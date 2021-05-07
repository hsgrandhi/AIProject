                                                               Project-World Models 
                                                  - Can agents learn inside of their own dreams?
                                                               - Submitted by:
                                                            Hemanth Grandhi- hg2315,
                                                            Deepankar Dixit - dd2943,
                                                            Divya Gupta- dg3483
                                                                 –May, 2021–
**Keywords:<br />**
Model  based  Reinforcement  learning  ,  Recurrent  neural  network  architecture  (MDN-RNN),  Variational  autoencoder  (VAE),  Controller,  Covariance  matrix  adaptation  evo-lution strategy (CMA-ES)

**Contents-: <br />**
1)  4-page report of the technique (a model-based RL)<br />
2)  Training V, M and C model<br />
3)  Issues faced while training above Agent<br />
4)  Results of training<br />
5)  A new approach that combines the VAE with a GAN<br />
6)  Problem with VAE-GAN implementation<br />
7)  Comparison between VAE and VAE-GAN<br />
8)  Conclusion<br />
9)  References<br />


**4-page report of the technique (a model-based RL)<br />**


Reinforcement learning is concerned with how to go about making decisions and taking sequential actions in a specific environment to maximize a reward. Backed by computing power, it can explore different strategies (or “policies” in the RL literature) much faster than we can. On other hand, lack of prior knowledge that humans bring to new situations and environments, it tends to explore many more policies than a human would before finding an optimal one.<br />

A model-based RL attempts to overcome the issue of a lack of prior knowledge by enabling the agent — whether this agent happens to be a robot in the real world, an avatar in a virtual one, or just a piece software that take actions — to construct a functional representation of its environment. Model-based RL potential imapct is enormous. As AI becomes more complex and adaptive — extending beyond a focus on classification and representation toward more human-centered capabilities — model-based RL will almost certainly play an essential role in shaping these frontiers. </br>

A model has a very specific meaning in reinforcement learning. It refers to the different dynamic states of an environment and how these states lead to a reward. Model-based RL entails constructing such models. It tends to emphasize on the planning part. By leveraging the information it’s learned about its environment, model-based RL can plan rather than just react, even simulating sequences of actions without having to directly perform them in the actual environment. It incorporates a model of the agent’s environment, specifically one that influences how the agent’s overall policy is determined. Model-free RL, conversely, forgoes this environmental information and only concerns itself with determining what action to take given a specific state. As a result model-free RL only tends to emphasize on learning. </br>

Advantages of using Model-based RL can be trasferable to other goals and task. Learning a single policy is good for one task but if we can predict the dynamics of environment, we can generalize those insights to multiple tasks. On the other hand disadvantage of model based RL is that we have to learn both the policy as well as model which gives two different source of approximation error. Also it is computationally demanding. </br>

Differance between model-based and model-free can be understood by real world analogy. In naviagtion problem, we keep track of all the routes we have taken to begin creating a map of the area. Map would be incomplete but will still help us to plan the course ahead of time to avoid certain neighborhood while still optimizing for most direct route. We can think of it as model based approch. Another option is we would simply keep track of locations we visited and actions we have taken, ignoring the details of routes. When ever we find ourself in location we visited, we favour the directional choice that lead to good outcome over to the direction that led to negative outcome. We will not have knowledge of next location we arrive following the decision but we have a simple procedure in place as to what action to be taken at specific location. This approch is what model-free RL takes. </br>

Our main focusing is model-based RL. Agent's model is inspired by our cognitive system. We as humans develop a mental model of the world based on the what we are able to  perceive  with  our  limit  senses. Our brain  only  learns  an  abstract  representation of both spatial and temporal aspects of the information. We can only remember an abstract description of the information. There are evidence that what we perceive at  any  given  moment  is  governed  by  our brain  future  predictions  based  on  internal model.  It is predicting future sensory data given our current motor actions.  We instinctively act on this predictive model and were able to perform reflexive action in face of danger without the need of consciously planning out the course of action.</br>

World models in simmilar fasion can be trained in an unsupervised manner to learn a compressed  spatial  and  temporal  representation  of  the  environment.   We  extract  the features  from  the  world  model. That  is used  as  input  to  agents.   We  train  these agents on very compact and simple policy that can solve the required task. We have visual sensory components that compresses what it sees into small representative code. Memory component that make predictions about future codes base on historical information. Decision making component that decides what actions to be taken based only on the representations created by its vision and memory component. These three components Vision (V), Memory(M) and Controler (C) work together closely. </br>

Variational autoencoder (VAE) is used as V component. Agent gets a high dimentional input observation at each time step from environment. These inputs are 2D image frames that is part of a vedio sequence. The V component learn abstarct, compressed representation of each observed input frame. It compresses each frame it receives at time step t into low dimensional latent vector z_t. These compressed representation can be used to created original representation. </br>

M component is called mixed density network combined with RNN. Role of M component is to predict the future. It is expected to produce predictions of z vectors that V models produce. Since we deal with stochastic environment, we train our RNN to output probability density function p(z) instead of deterministic predictions of z.  p(z) is mixture of Gaussian distributions. Given current and past information, we can train our RNN to output probability distribution of next latent vector z_t+1. More specifically, RNN will produce model P(z_t+1 | a_t, z_t, h_t), where a_t is action taken at time t, h_t is hidden state of RNN at time t. We control model uncertainty by adjusting temprature parameter. </br>

We use recurrent neural networks to implement these powerful predictive models. Large RNNs are highly expressive models that can learn rich spatial and temporal representations of data. However RL algorithms is often bottlenecked by the credit assignent problem. It makes harder for traditional RL algorithms to learn millions of weights of a large model. Hence in practice, smaller networks are used to iterate faster to a good policy during training. <br/>

Controler component determine the course of action taken in order to maximize the expected cumulative reward of the agent during rollout of the environment. It is made simple and small deliberately and made to train separately from V and M so that most of our agents complexity lies resides in world models( V and M). It is simple single layer linear model that maps z_t and h_t directly with a_t at each step. </br>
                              **a_t= W_c [z_t h_t] + B_c </br>**
where w_c and B_c are weight matrix and bias vector that concatenate input vector [z_t h_t] with output a_t.</br>

Together V, M and C works as follows. Raw observation first processed by V at each time step t to produce z_t. The latent vector z_t concatenated with M hidden state h_t is passed to C. C will output an action vector a_t for motor cotrol which effects the environment. M will take current vector z_t and action vector a_t as input to update its hidden state to produce h_t+1. After running this controller c will return the cumulative_reward during the rollout. </br>

We can train large and sophisticated models efficiently, provided we define a well behaved, differential loss functions. Most of the model complexity and model parameters would reside in V and M. Number of parameters in C are minimal in comparison to M and V. This allow us to explore more unconvetional ways to train C example, even using evolution strategies (ES) to tackle more challeging RL tasks where credit assignment problem is difficult. Covariance matrix adaptation evolution strategy (CMA-ES) is used to optimize the parameter C. It works well for solution spaces of up to a few thousand parameters. </br>

In car racing project, we are training large neural network to tackle RL tasks. Agent is divided into large world model and small controller model. We first train a large neural network to learn model of Agents world in an unsupervised manner. Then we train small controller model to learn to perform task using world model. Training algorithm of small controller, focus on credit assignment problem on small search space while not sacrificing capacity and expressiveness via large world models. Training the agent through the lens of its world model, It was shown here that agent can learn highly compact policy to perform the task.</br>

Though we learn a model of the RL environment using existing model based Rl, we still train on actual environment. Here in car racing experiment, an actual RL environment is replaced with the generated one. We trained our controler inside the environment generated	by its own internal world model and then transfer this policy back into actual environment. We can produce the sample of probability destribution of z_t+1 given the current state and the use those sample as real observations to produce a virtual environment. Adjustments are made in the temprature parameter of the internal world model to control the amount of uncertaininty of the generated environments. This helps to overcome the problem of agents exploiting the imperfections of generated environment. We can also demonstrate that if we can successfully train our agents controller inside of a noisier and more uncertain version of its generated environment, Agent will definately thrive in the original, cleaner environment </br>

Using M model to generate a virtual dream environment can give access to all the hidden state of M to controller. Therefore our agent can efficiently explore ways to directly manipulate the hidden states of the agent engine in its quest to maximize its expected cumulative reward. This is a down side of learning a policy inside a learned dynamics model. Our agent can easily find an adversarial policy that can fool our dynamics model – it’ll find a policy that looks good under our dynamics model, but will fail in the actual environment. This is the reason we do not use those models to replace actual environment. Recent solutions combines model based approch with traditional model free RL training by first intialzing the policy network with learned policy but subsequently, reply on model free methods to fine tune learned policy in actual environment.</br>

In car racing experiment, RNN M is used to predict and plan ahead step-by-step and we have used evolution to optimize C. Dynamic MDN-RNN is used as M model to make even more harder for C to exploit its deficiencies. MDN-RNN models the distribution of possible outcome in actual environment rather than merely predicting a deterministic future. Even though actual environment is deterministic, it would in effect approximate it as a stochastic environment. This gives us the advantage of training C in a stochastic version of any environment. We can control the tradeoff between realism and exploitability by adjusting the temperature hyper-parameter. It also makes it easier to model the logic behind a more complicated environment with discrete random states. </br>

In any difficult environment, where some parts of world are made available to the agent only when it learns how to strategically navigate through the world. We need iterative training procedures to perform complicated tasks. Here agent is able to explore world and constantly collect new observations to improve and refine its world model over time. An iterative training procedure is as follows. </br>
1) Initialize M, C with random model parameters.</br>
2) Rollout to actual environment N times. Save all actions at and observations xt during rollouts to storage.</br>
3) Train M to model P(x_t+1, r_t+1, a_t+1, d_t+1| xt, at, ht) and train C to optimize expected rewards inside of M.</br>
4) Go back to (2) if task has not been completed.</br>

In present approch, MDN-RNN that models a probability distribution for the next frame. If it does the poor job, it means that it encounters the part of world that it is not familiar with. In this case, We can reuse M training loss function to encourage curiosity. We flipped the sign of M's loss function in actual environment to encourage the agent to explore the parts of the world that it is not familar with. New data collected improves the world model. Iterative training  procedure requires M model to also predict the action and reward for the next step. This is useful for more difficult tasks. For instance we require world model to imitate controller that has already learned walking. Once world model absorb the skill of walking, controller can rely on those and focus on learning more higher level skill. </br>

Above we have discussed above about the reinforcement learning, model-based architecture. We have demonstrated the possibility of training an agent entirely inside simulated latent space dream world. We have discussed about the benifits of implementing world models as fully differential recurrent computational graph which also means using back propagation algorithm to train our agent in dream is to fine tune its policy to maximize the objective function. Further we can see the choice of using and training VAE for V model as a standalone model has its own limitations as it can encodes parts of observation that are not relevent to the task. We can train it with M model that predicts rewards and VAE can learn t focus on learning task relevent areas of the image. But downside of using this technique is that we can not reuse VAE for new tasks without retraining. Also we can see LSTM based models are not capable of storing all recorded information inside its weighted connections. It suffers from catastrophic forgetting. Future work is needed in replacing small MDN-RNN network with higher capacity models. Recent one is One Big Net approch that collapse C and M into single network and use power play like behavioural replay to avoid forgetting old predictions and control skill when learning new one. 


**Training V, M and C model<br />**
A predictive world model is used extract useful representations of space and time. Using these features as inputs of a controller, we can train a compact and minimal controller to perform a continuous control task such as learning to drive from pixel inputs from top-down car racing environment. Reward is -0.1 every frame and + 1000/N for every track tile visited, where N is total number of tiles in track. For example, if you finished in 732 frames, you reward is 1000-0.1*732= 926.8 points. For good performance, we need to get 900+ points consistently. Track is random every episode. Episode finishes when all tiles are visited. If car go far off the track, then it will get -100 and die. Agent conrol three continuous actions: steering left/right, acceleration and brake. </br>

First, collect a dataset of 2000 random rollouts of the environment. We have an agent acting randomly to explore the environment multiple times and record the random actions a_t taken and the resulting observtions from the environment. 01_generate_data.py is used to collect this data in folder data\rollout. </br>

We use this dataset to train V to learn a latent space of each frame observed. We encode each frame in low dimentional latent vector z_t by minimizing the difference between a given frame and the reconstructed version of the frame produced by the decoder from z. 02_train_vae.py is used to train over 100 episodes. </br>

We can now use our trained V model to pre-process each frame at time t into z_t to train our M model. Pre-processed data, along with the recorded random actions a_t taken, our MDN-RNN can now be trained to model P(z_t+1 | a_t, z_t, h_t) as a mixture of gaussians. 03_generate_rnn_data.py prepossed data and store in data/series folder. We then train our M component using 04_train_rnn.py </br>

World model (V and M) does not have knowledge of actual reward signals from the environment. Its task is simply to compress and predict the sequence of image frames observed. Controler function have access to reward information from the environment. CMA-ES evolutionary algorithm is well suited to optimize parameters inside linear controller model. To train our controller we have use 05_train_controller.py.</br>

To summarize the Car Racing experiment, below are the steps taken:</br>
1. Collect 2000 rollouts from a random policy.</br>
2. Train VAE (V) to encode frames into z 2 R32.</br>
3. Train MDN-RNN (M) to model P(zt+1 j at; zt; ht).</br>
4. Define Controller (C) as at = Wc [zt ht] + bc.</br>
5. Use CMA-ES to solve for a Wc and bc that maximizes the expected cumulative reward.</br>

