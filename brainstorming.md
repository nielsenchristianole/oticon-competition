
### Models
---
In this project we have choosen to implement and traing 4 different kinds of models to compare them against eachother. The models implemented are:
* Feed Forward Neural Network (FFNN)
* Convolutional Neural Network (CNN)
* Long-Short Term Memory (LSTM)
* Encoder-only Transform (BART)




### Unbalanced traning set
---
As it often happens, collecting data for certain classes are much easier than others. In this project the class distribution is heavily skewed which may introduce a bias in the training. Even though the validation score may be high, having a hearing aid that can not detect Sirens, due to lack of training data, would be critical and in some cases dangerous. To tackle this problem we use the weighted Negative Log Likelihood as objective function for our training algorithms. 

[NLLLOSS](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)




### Network pruning
---
As mentioned one of the big problems with running neural networks on small hearing aid devices are the limited memory and battery. One solution for reducing the computational cost of a forward pass is to reduce the number of parameters. Recent development with transformers (1)(2)(3)(4) has shown how the complexity of neural models has increased by stacking more layers ontop of each other. This means, in broad terms, that introducing more parameters will create a more flexible and advanced network as long as regularization methods such as dropout and weightdecay are introduced.  
To allow high complexity with a moderately low number of parameters pruning can be used. Methods such as Optimal Brian Damage (5), The Lottery Ticket Hypothesis (6), and Synaptic Flow (7) has shown how the number of paramters can be reduced by the use of Pruning.  
Easy to use pre-existing methods of pruning are already avaliable in PyTorch (7) and will therefore be implemented and tested if the time allowes for it.

### References
---

(1) GPT2, https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf  
(2) GPT3, https://arxiv.org/pdf/2005.14165.pdf  
(3) GPT3.5, https://platform.openai.com/docs/model-index-for-researchers  
(4) GPT4, https://arxiv.org/pdf/2303.08774.pdf  
(5) Optimal Brian Damage, https://proceedings.neurips.cc/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf  
(6) The Lottery Ticket Hypothesis, https://arxiv.org/pdf/1803.03635.pdf  
(7) Synaptic Flow, https://arxiv.org/pdf/2006.05467.pdf  
(8) Pytorch pruning, https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
