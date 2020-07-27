import numpy as np
import tensorflow as tf
from Figure_3_and_S7_e_prop_tutorials.models import EligALIF, exp_convolve, shift_by_one_time_step
from Figure_4_and_5_ATARI.alif_eligibility_propagation import CustomALIF


def val_compare(name, val_a, val_b):
    if val_a is None or val_b is None:
        print('  There is no {}.'.format(name))
    else:
        error = ((val_a - val_b) / np.max(val_a)) ** 2
        max_error = np.max(error)
        print("  Maximum element wise errors({}): {}".format(name, max_error))
    
    
def relative_error(name_list, np_tensors_a, np_tensors_b):
    for name in name_list:
        val_compare(name, np_tensors_a[name], np_tensors_b[name])
            
            
# 1. Let's define some parameters
n_in = 3
n_LIF = 10
n_ALIF = 10
n_rec = n_ALIF + n_LIF

dt = 1  # ms
tau_v = 20  # ms
tau_a = 500  # ms
T = 50  # ms
f0 = 100  # Hz

thr = 0.62
beta = 0.07 * np.concatenate([np.zeros(n_LIF),np.ones(n_ALIF)])
dampening_factor = 0.3
n_ref = 2

w_out = tf.random_normal(shape=[n_rec, 1], seed=1)
decay_out = tf.exp(-1/20)

# 2. Define the network model and the inputs
cell = EligALIF(n_in=n_in, n_rec=n_LIF + n_ALIF, tau=tau_v, beta=beta, thr=thr,
                dt=dt, tau_adaptation=tau_a, dampening_factor=dampening_factor,
                stop_z_gradients=False,  # here it computes the BPTT gradients, set it to True to compute instead e-prop with auto-diff
                n_refractory=n_ref)

cell_auto = EligALIF(n_in=n_in, n_rec=n_LIF + n_ALIF, tau=tau_v, beta=beta, thr=thr,
                     dt=dt, tau_adaptation=tau_a, dampening_factor=dampening_factor,
                     stop_z_gradients=True,
                     n_refractory=n_ref)

cell_CustomALIF = CustomALIF(n_in=n_in, n_rec=n_LIF + n_ALIF, tau=tau_v, beta=beta, thr=thr,
                             dt=dt, tau_adaptation=tau_a, dampening_factor=dampening_factor,
                             stop_gradients=False,
                             n_refractory=n_ref)

cell_CustomALIF_auto = CustomALIF(n_in=n_in, n_rec=n_LIF + n_ALIF, tau=tau_v, beta=beta, thr=thr,
                                  dt=dt, tau_adaptation=tau_a, dampening_factor=dampening_factor,
                                  stop_gradients=True,
                                  n_refractory=n_ref)

inputs = tf.random_uniform(shape=[1, T, n_in], seed=1) < f0 * dt / 1000
inputs = tf.cast(inputs, tf.float32)
y_target = tf.random_normal(shape=[1, T, 1], seed=1)

# 3. Simulate the network,
# Using a for loop instead of using tf.nn.dynamic_rnn(...) is not efficient
# belows allows us to compute the true learning signals: dE/dz (total derivative)
# with auto-diff to perform the numerical verification
spikes = []
spikes_auto = []
spikes_CustomALIF = []
spikes_CustomALIF_auto = []

hidden_states = []
hidden_states_auto = []
hidden_states_CustomALIF = []
hidden_states_CustomALIF_auto = []

state = cell.zero_state(1, tf.float32, n_rec=n_rec)
state_auto = cell_auto.zero_state(1, tf.float32, n_rec=n_rec)
state_CustomALIF = cell_CustomALIF.zero_state(1, tf.float32, n_rec=n_rec)
state_CustomALIF_auto = cell_CustomALIF_auto.zero_state(1, tf.float32, n_rec=n_rec)

for t in range(T):
    outs, state = cell(inputs[:, t], state)
    outs_auto, state_auto = cell_auto(inputs[:, t], state_auto)
    outs_CustomALIF, state_CustomALIF = cell_CustomALIF(inputs[:, t], state_CustomALIF)
    outs_CustomALIF_auto, state_CustomALIF_auto = cell_CustomALIF_auto(inputs[:, t], state_CustomALIF_auto)

    spikes.append(outs[0])
    hidden_states.append(outs[1])

    spikes_auto.append(outs_auto[0])
    hidden_states_auto.append(outs_auto[1])

    spikes_CustomALIF.append(outs_CustomALIF[0])
    hidden_states_CustomALIF.append(outs_CustomALIF[1])

    spikes_CustomALIF_auto.append(outs_CustomALIF_auto[0])
    hidden_states_CustomALIF_auto.append(outs_CustomALIF_auto[1])

# 4. Compute the true learning signal: dE/dz (total derivative) for an arbitrary loss function
# (here a regression with a random signal)

spikes_stacked = tf.stack(spikes, axis=1)
spikes_stacked_auto = tf.stack(spikes_auto, axis=1)
spikes_stacked_CustomALIF = tf.stack(spikes_CustomALIF, axis=1)
spikes_stacked_CustomALIF_auto = tf.stack(spikes_CustomALIF_auto, axis=1)

z_filtered = exp_convolve(spikes_stacked, decay_out)
y_out = tf.matmul(z_filtered, w_out)

z_filtered_auto = exp_convolve(spikes_stacked_auto, decay_out)
y_out_auto = tf.matmul(z_filtered_auto, w_out)

z_filtered_CustomALIF = exp_convolve(spikes_stacked_CustomALIF, decay_out)
y_out_CustomALIF = tf.matmul(z_filtered_CustomALIF, w_out)

z_filtered_CustomALIF_auto = exp_convolve(spikes_stacked_CustomALIF_auto, decay_out)
y_out_CustomALIF_auto = tf.matmul(z_filtered_CustomALIF_auto, w_out)

loss = 0.5 * tf.reduce_sum(tf.square(y_out - y_target))
loss_auto = 0.5 * tf.reduce_sum(tf.square(y_out_auto - y_target))
loss_CustomALIF = 0.5 * tf.reduce_sum(tf.square(y_out_CustomALIF - y_target))
loss_CustomALIF_auto = 0.5 * tf.reduce_sum(tf.square(y_out_CustomALIF_auto - y_target))

# This defines the true learning signal: dE/dz (total derivative)
dE_dz = tf.stack(tf.gradients(loss, [spikes[t] for t in range(T)]), axis=1)
dE_dz_auto = tf.stack(tf.gradients(loss_auto, [spikes_auto[t] for t in range(T)]), axis=1)

# Stack the lists as tensors (second dimension is time)
# - spikes and learning signals will have shape: [n_batch, n_time , n_neuron]
# - eligibility traces will have shape: [n_batch, n_time , n_neuron, n_neuron]
hidden_states = tf.stack(hidden_states, axis=1)
hidden_states_auto = tf.stack(hidden_states_auto, axis=1)

# 5. Compute the eligibility traces for ALIFs using formula (25)

# for cell
v_scaled = cell.compute_v_relative_to_threshold_values(hidden_states)
spikes_last_time_step = shift_by_one_time_step(spikes_stacked)
eligibility_traces, _, _, _ = cell.compute_eligibility_traces(v_scaled, spikes_last_time_step, spikes_stacked, True)
gradients_hardcode = tf.einsum('btj,btij->ij', dE_dz, eligibility_traces)

# for cell_auto
v_scaled_auto = cell_auto.compute_v_relative_to_threshold_values(hidden_states_auto)
spikes_last_time_step_auto = shift_by_one_time_step(spikes_stacked_auto)
eligibility_traces_auto, _, _, _ = cell_auto.compute_eligibility_traces(v_scaled_auto, spikes_last_time_step_auto, spikes_stacked_auto, True)
gradients_hardcode_auto = tf.einsum('btj,btij->ij', dE_dz_auto, eligibility_traces_auto)


# 7. Compute the gradients given by auto-diff
gradients_BPTT = tf.gradients(loss, cell.w_rec_var)[0]
gradients_AUTO = tf.gradients(loss_auto, cell_auto.w_rec_var)[0]

gradients_CustomALIF_BPTT = tf.gradients(loss_CustomALIF, cell_CustomALIF.w_rec_var)[0]
gradients_CustomALIF_AUTO = tf.gradients(loss_CustomALIF_auto, cell_CustomALIF_auto.w_rec_var)[0]

# 8. Start the tensorflow session and compute numerical verification.
# (until now we only built a computational graph, no simulation has been performed)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


tf_tensors = {'inputs': inputs,
              'target': y_target,
              'w_in': cell.w_in_var,
              'w_rec': cell.w_rec_var,
              'w_out': w_out,
              'out': y_out,
              'spikes': spikes_stacked,
              'gradients_hardcode': gradients_hardcode,
              'gradients_autodiff': gradients_BPTT,
              'eligibility_traces': eligibility_traces,
              'learning_signals': dE_dz}

tf_tensors_auto = {'inputs': inputs,
                   'target': y_target,
                   'w_in': cell_auto.w_in_var,
                   'w_rec': cell_auto.w_rec_var,
                   'w_out': w_out,
                   'out': y_out_auto,
                   'spikes': spikes_stacked_auto,
                   'gradients_hardcode': gradients_hardcode_auto,
                   'gradients_autodiff': gradients_AUTO,
                   'eligibility_traces': eligibility_traces_auto,
                   'learning_signals': dE_dz_auto}

tf_tensors_CustomALIF = {'inputs': inputs,
                          'target': y_target,
                          'w_in': cell_CustomALIF.w_in_var,
                          'w_rec': cell_CustomALIF.w_rec_var,
                          'w_out': w_out,
                          'out': y_out_CustomALIF,
                          'spikes': spikes_stacked_CustomALIF,
                          'gradients_autodiff': gradients_CustomALIF_BPTT}

tf_tensors_CustomALIF_auto = {'inputs': inputs,
                              'target': y_target,
                              'w_in': cell_CustomALIF_auto.w_in_var,
                              'w_rec': cell_CustomALIF_auto.w_rec_var,
                              'w_out': w_out,
                              'out': y_out_CustomALIF_auto,
                              'spikes': spikes_stacked_CustomALIF_auto,
                              'gradients_autodiff': gradients_CustomALIF_AUTO}

np_tensors, np_tensors_auto, np_tensors_CustomALIF, np_tensors_CustomALIF_auto = sess.run([tf_tensors, tf_tensors_auto, tf_tensors_CustomALIF, tf_tensors_CustomALIF_auto])

# Compute the relative error:
np_tensors_CustomALIF['gradients_hardcode'] = None
np_tensors_CustomALIF_auto['gradients_hardcode'] = None
name_list = ['inputs', 'target', 'w_in', 'w_rec', 'w_out', 'spikes', 'out', 'gradients_hardcode', 'gradients_autodiff']

# S1:  Compare gradients from BPTT and eprop-autodiff based EligALIF model
print('S1: EligALIF model BPTT vs. eprop-autodiff')
relative_error(name_list, np_tensors, np_tensors_auto)
print('######################################\n\n')

# S2:  Compare gradients from autodiff and hardcode based EligALIF model
print('S2: EligALIF model autodiff vs. hardcode')
print('  BPTT vs. eprop-hardcode')
val_compare('gradients BPTT vs. eprop-hardcode', np_tensors['gradients_hardcode'], np_tensors['gradients_autodiff'])
print('  eprop-autodiff vs. eprop-hardcode')
val_compare('gradients eprop-autodiff vs. eprop-hardcode', np_tensors_auto['gradients_hardcode'], np_tensors_auto['gradients_autodiff'])
print('######################################\n\n')

# S3:  Compare gradients from BPTT and eprop-autodiff based CustomALIF model
print('S3: CustomALIF model BPTT vs. eprop-autodiff')
relative_error(name_list, np_tensors_CustomALIF, np_tensors_CustomALIF_auto)
print('######################################\n\n')



