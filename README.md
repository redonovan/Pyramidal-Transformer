# Pyramidal-Transformer
<B>Research</B>

<a href="https://arxiv.org/abs/1508.01211">Listen, Attend and Spell</a> was published in 2015, before the invention of the Transformer, and so it is natural to wonder whether using a Transformer encoder in place of the bidirectional LSTM encoder in the Listener would be helpful; doubly so since I read <a href="https://arxiv.org/abs/1910.09799">Wang, et al. 2020</a> which found Transformer encoders to be useful in Hybrid systems.

<P>

  My code for the Transformer version of the system is in <a href="TransformAttendSpell.py">TransformAttendSpell.py</a>.  The absence of a pyramidal structure led to a larger memory requirement during training necessitating a batch_size of 1.  Gradient norm clipping was introduced to stabilise training.  Training was also slower, with only 1 epoch of training being possible in the 8-9 hrs I allow for training.  The Tensorboard training curve is in <a href="TransformTensorBoard.png">TransformTensorBoard.png</a>.  Validation data next-character prediction accuracy at the end of training was 0.4751, compared to my Listen Attend and Spell results of 0.4929 and 0.5278 after 1 epoch and 8-9 hours of training respectively.  Both systems had approximately the same number of weights, about 8.5m, in their Listeners.

<P>

  It was noted in the original Listen, Attend and Spell paper that the pyramid structure was necessary for the model to converge in reasonable timeframes.  It occurred to me therefore that one reason my Transformer version of LAS described above might not have beaten my original BLSTM version is that it lacks this pyramidal structure.  I therefore implemented a pyramidal version in which each Transformer layer reshapes the output of its self-attention sub-layer to halve the number of timesteps, doubling the dimensionality, and then uses the pointwise feed-forward sub-layer to reduce the dimensionality back to its original value.  My code is in <a href="PyTranAttendSpell.py">PyTranAttendSpell.py</a>.  This system could be run with a batch_size of 4 but nevertheless benefited from gradient norm clipping.  This system was trained for 3 epochs over 9 hours.  The Tensorboard training curve is in <a href="PyTranTensorBoard.png">PyTranTensorBoard.png</a>.  Validation data next-character prediction results were 0.4844 and 0.5108 accuracy after 1 epoch and 9 hours respectively.  In this system the Transformer pointwise feed-forward network intermediate dimension was reduced to 1280 so as to have approximately the same total number of weights (8.3m) in the Listener as the other systems.

<P>

  Thus neither my Transformer system nor my Pyramidal Transformer system beat my original LAS system, though the results were all fairly similar.  The next thing I had intended to work on was dynamic merging/masking, hopefully "learning to merge/mask".  However, even my first functioning attempt, which used a prescriptive technique to dynamically mask some acoustic timesteps, also seemed to give similar results.  By now my suspicion that my systems might be ignoring the acoustics completely - something mentioned as a problem in the original LAS paper - was very high.  I ran my dynamic masking system with settings that made it mask 97% of acoustic timesteps, and it still achieved a validation accuracy of 0.4817 after 1 epoch!

<P>

  I decided to investigate further by plotting decoder attention weights between characters and acoustics.  The Attention layer in TensorFlow 2.3.0 does not have an option to return attention weights but it can be tricked into doing so by passing an identity matrix as the value argument.  Doing so yields the following for my Transformer system:  <a href="TransformAttentionWeights.png">TransformAttentionWeights.png</a>  and the following for my Pyramidal Transformer system: <a href="PyTranAttentionWeights.png">PyTranAttentionWeights.png</a>.  In these cases analysis shows that the Listener representations h(u) are virtually identical at every timestep.  That is, the Transformer Listeners have learned nothing, and completely average out the input logmels during inference.  I then plotted attention weights for my LAS system: <a href="LASAttentionWeights.png">LASAttentionWeights.png</a>.  In this case every decoder step attends to the same acoustic representation timestep.  Analysis shows that in the LAS system the h(u) values are at least different over timesteps.

<P>

  It is now clear that none of my LAS-based systems are paying attention to the acoustics.  This problem was mentioned in the LAS paper, but the solution given, using the attention mechanism, is already in place in my system.  Possibly I have this problem because I have so much less training data than was used in the paper.  Possibly it is because my training sentences are too long; Figure 4 in the paper shows a data distribution whose mode is only 3 words long, mine is 37.  Possible solutions include curriculum learning, location-sensitive attention, and perhaps introducing information bottlenecks into the decoder as used in <a href="https://arxiv.org/abs/1712.05884">Tacotron 2</a>.

TODO : this page and its systems need revisiting in the light of improvements made to my LAS system.




