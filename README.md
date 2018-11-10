# Walk with a Prosthetic Leg (work in progress)

This is the project for the CrowdAI competition on [NIPS 2018: AI for Prosthetics Challenge](https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge), to develop a controller for a physiologically-based human model with a prosthetic leg to walk and run.

The model is based on the [distributed DDPG implementation of OpenAI baselines](https://github.com/openai/baselines/tree/master/baselines/ddpg). 
* **Actor**: to learn high level joint angles and low level muscle excitations. This is inspired by human control of motions. We have high level controls to consciously move our limbs to certain positions or angles, while the low level controls (as the result of the evolution) take care of the specific muscle activation. 
* **Critic**: to predict Q value and to model the kinematics (angles of all joints and velocity of key body parts) of the next state. Q value is the accumulation of reward, should it model the kinematic function of the corresponding future states?


![](problem.png) 
