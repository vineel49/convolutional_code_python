# convolutional coded QPSK over AWGN
# Generator matrix G(D)=[1 (1+D^2)/(1+D+D^2)]
import time
import numpy as np
frame_size = 1024 #frame size
SNR_dB = 6 # SNR per bit (dB)
decoding_delay = 20 # decoding delay of the Viterbi algorithm
NP = [1,0,1] # numerator polynomial of the G(D)
DP = [1,1,1] # denominator polynomial of the G(D)
SNR = 10**(0.1*SNR_dB) # SNR in linear scale
noise_var_1D = 1/SNR # 1D AWGN variance
sim_runs = 1e2 # simulation runs

# generator polynomial of the encoder using long division method
gen_poly = np.zeros(frame_size)
for i1 in range(frame_size):
    gen_poly[i1]=NP[0]
    temp=np.logical_xor(NP,np.dot(DP,NP[0]))
    NP = np.append(temp[1:],0)
#---------------------------------------------------------------
# Trellis for the RSC encoder
num_states = 4 # number of states
Prev_State = np.array([[0,1],[3,2],[1,0],[2,3]]) # previous state
Prev_Ip = np.array([[0,1],[0,1],[0,1],[0,1]]) # previous ip
Outputs_prev = np.array([[0,3],[1,2],[0,3],[1,2]]) # branch indices
Prev_State_Ftnd = Prev_State.flatten() # flattened
Prev_Ip_Ftnd = Prev_Ip.flatten() # flattened
#---------------------------------------------------------------
C_BER = 0 # channel errors (total)
start_time = time.time()
for i in range(int(sim_runs)):

    # source
    a=np.random.randint(2,size=frame_size)

    # convolutional encoder
    b=np.zeros(2*frame_size) # encoder output intialization
    b[::2]=a # systematic bit
    temp = np.remainder(np.convolve(gen_poly,a),2)
    b[1::2]=temp[0:frame_size] # parity bit

    # QPSK mapping
    s = 1-2*b[0::2] + 1j*(1-2*b[1::2])

    # AWGN
    awgn = np.random.normal(0,np.sqrt(noise_var_1D),frame_size)+1j*np.random.normal(0,np.sqrt(noise_var_1D),frame_size)

    # channel output
    chan_op = s+awgn

    # Receiver
    # branch metrics for the Viterbi algorithm
    QPSK_SYM = np.zeros((4,frame_size),dtype=complex)
    QPSK_SYM[0] = (1+1j)*np.ones((1,frame_size))
    QPSK_SYM[1] = (1-1j)*np.ones((1,frame_size))
    QPSK_SYM[2] = (-1+1j)*np.ones((1,frame_size))
    QPSK_SYM[3] = (-1-1j)*np.ones((1,frame_size))

    branch_metric = np.zeros((4,frame_size))
    branch_metric[0] = np.power(np.abs(chan_op-QPSK_SYM[0]),2)
    branch_metric[1] = np.power(np.abs(chan_op-QPSK_SYM[1]),2)
    branch_metric[2] = np.power(np.abs(chan_op-QPSK_SYM[2]),2)
    branch_metric[3] = np.power(np.abs(chan_op-QPSK_SYM[3]),2)

    # Soft-input, Hard-output Viterbi algorithm
    # general initialization
    ip = 0 
    dec_a = np.zeros(frame_size-decoding_delay)
    survivor_node = np.zeros((num_states,frame_size))
    survivor_ip = np.zeros((num_states,frame_size))
    path_metric = np.zeros((num_states,frame_size+1))
    #
    for sym_cnt in range(frame_size):
     temp1 = path_metric[Prev_State[:,0],sym_cnt]+branch_metric[Outputs_prev[:,0],sym_cnt]
     temp2 = path_metric[Prev_State[:,1],sym_cnt]+branch_metric[Outputs_prev[:,1],sym_cnt]
     conc_mat = np.concatenate((temp1.reshape(num_states,1),temp2.reshape(num_states,1)),axis=1)
     path_metric[:,sym_cnt+1] = np.amin(conc_mat,axis=1)
     index = np.argmin(conc_mat,axis=1)
     lin_indexing = np.array([0,2,4,6]) # linear indexing
     survivor_node[:,sym_cnt] = Prev_State_Ftnd[index+lin_indexing]
     survivor_ip[:,sym_cnt] = Prev_Ip_Ftnd[index+lin_indexing]
     if (sym_cnt>=decoding_delay-1):
         trace_bf = np.argmin(path_metric[:,sym_cnt+1],axis=0)
         for bk_cnt in range(decoding_delay+1):
             ip = survivor_ip[trace_bf,sym_cnt-bk_cnt]
             trace_bf = survivor_node[trace_bf,sym_cnt-bk_cnt].astype(int)
         dec_a[sym_cnt-decoding_delay]=ip
    dec_a = dec_a.astype(int)
    # calculating total number of bit errors
    C_BER = C_BER + np.count_nonzero(a[0:frame_size-decoding_delay]-dec_a)
#----------------------------------------------------------------------------
elapsed_time = time.time() - start_time
# calculating Bit error rate
BER = C_BER/((frame_size-decoding_delay)*sim_runs)
print("elapsed time is: ",elapsed_time ,"seconds")
print("Bit error rate is: ",BER)
