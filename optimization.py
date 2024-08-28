from PCRblocker import optimization

params = {'Btot' : 10,  # total concentration of the blocker strands [uM]
          'blocker length': 14, # length of the blocker strand [nt]
          'temperature': 60,   # annealing temperature in the PCR cycle [degree in Celcius]
          'calib_a':0.6185290898147885, 'calib_b':-4.3830120235322925,  # parameters for the calibration of Kij (see the paper)
          'max iteration count': 1000000,  # the maximum number of iterations to solve the replicator equation 
          'error tolerance': 0.0001, # the iteration stops when this tolerance is reached.
          'report convergence': False,  # wheteher the information on the convergence is shown during iterations
          'p': 0.12079, 'q': 2.59900104, 'r': 2.1268991, 's': 0.49584249, 'u': 15.65649238  # the fitting parameters for \bar\epsilon_i (sse the paper)
          }


# This is an example using the sequences in Takahashi et al., bioRxiv (2024).
# https://www.biorxiv.org/content/10.1101/2024.04.19.590219v1

template = "CTTCTCTCTTCAGATTGACGACGCATATCTCAAGG" # The template sequence in 5' -> 3' order. The primer binding site should be located at the 3' end.
# Additional sequence is necessary at 5' end so that (length of the template) >= primer_length + 'blocker length'/2.
# Here, primer_legnth is given as the length of W (all W sequences should have the same length)


# Scenario (i) of Takahashi et al., bioRxiv (2024).
W = ["TGACG ACGCA TATCA CAAGG","TGACG ACGCT TATCT CAAGG", "TGTCG ACGCA TATCT CAAGG"] # 5' -> 3'. The wrong template sequences. Only the sequences of the primer binding sites are specified.


# The primer sequence is given as the sequence complementary to the 3' end of template with the length that equals to the length of W.

p = [1/len(W)]*len(W)

optimization(params, template, W, p, ifMean = True)
optimization(params, template, W, p, ifMean = False)

# Scenario (ii) of Takahashi et al., bioRxiv (2024).
W = ["TGACG ACGCT TATCT CAAGG","TGACG ACGCC TATCT CAAGG", "TGACG ACGCG TATCT CAAGG"] # 5' -> 3'

optimization(params, template, W, p, ifMean = True)
optimization(params, template, W, p, ifMean = False)