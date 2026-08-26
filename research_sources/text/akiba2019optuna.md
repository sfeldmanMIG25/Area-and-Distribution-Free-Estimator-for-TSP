# Optuna: A next-generation hyperparameter optimization framework

> Citation key: `akiba2019optuna`  
> DOI: 10.1145/3292500.3330701  
> Download URL: https://arxiv.org/pdf/1907.10902  
> SHA-256: `a1ec86ba21a3f2e7bb774c00f8b108dc955993f39feed72f5d8ee3910b6d8a51`  

---

                                             Optuna: A Next-generation Hyperparameter Optimization Framework
                                                                                          Preprint, compiled July 26, 2019

                                                                          ∗
                                                         Takuya Akiba1 , Shotaro Sano1 , Toshihiko Yanase1 , Takeru Ohta1 , and Masanori Koyama1
                                                                                              1
                                                                                                  Preferred Networks, Inc.


                                                                                                       Abstract
                                              The purpose of this study is to introduce new design-criteria for next-generation hyperparameter optimization software.
                                              The criteria we propose include (1) define-by-run API that allows users to construct the parameter search space
                                              dynamically, (2) efficient implementation of both searching and pruning strategies, and (3) easy-to-setup, versatile
                                              architecture that can be deployed for various purposes, ranging from scalable distributed computing to light-weight
arXiv:1907.10902v1 [cs.LG] 25 Jul 2019




                                              experiment conducted via interactive interface. In order to prove our point, we will introduce Optuna, an optimization
                                              software which is a culmination of our effort in the development of a next generation optimization software. As an
                                              optimization software designed with define-by-run principle, Optuna is particularly the first of its kind. We will present
                                              the design-techniques that became necessary in the development of the software that meets the above criteria, and
                                              demonstrate the power of our new design through experimental results and real world applications. Our software is
                                              available under the MIT license (https://github.com/pfnet/optuna/).

                                         1    Introduction                                                    be in vain. Secondly, many existing frameworks do not feature
                                                                                                              efficient pruning strategy, when in fact both parameter searching
                                         Hyperparameter search is one of the most cumbersome tasks            strategy and performance estimation strategy are important for
                                         in machine learning projects. The complexity of deep learning        high-performance optimization under limited resource availabil-
                                         method is growing with its popularity, and the framework of effi-    ity [12][5][7]. Finally, in order to accommodate with a variety
                                         cient automatic hyperparameter tuning is in higher demand than       of models in a variety of situations, the architecture shall be able
                                         ever. Hyperparameter optimization softwares such as Hyper-           to handle both small and large scale experiments with minimum




                                                                                                                                                                                     Preprint doi
                                         opt [1], Spearmint [2], SMAC [3], Autotune [4], and Vizier [5]       setup requirements. If possible, architecture shall be installable
                                         were all developed in order to meet this need.                       with a single command as well, and it shall be designed as
                                         The choice of the parameter-sampling algorithms varies across        an open source software so that it can continuously incorporate
                                         frameworks. Spearmint [2] and GPyOpt use Gaussian Pro-               newest species of optimization methods by interacting with open
                                         cesses, and Hyperopt [1] employs tree-structured Parzen estima-      source community.
                                         tor (TPE) [6]. Hutter et al. proposed SMAC [3] that uses random In order to address these concerns, we propose to introduce the
                                         forests. Recent frameworks such as Google Vizier [5], Katib following new design criteria for next-generation optimization
                                         and Tune [7] also support pruning algorithms, which monitor framework:
                                         the intermediate result of each trial and kills the unpromising
                                         trials prematurely in order to speed up the exploration. There is     • Define-by-run programming that allows the user to dynami-
                                         an active research field for the pruning algorithm in hyperparam-       cally construct the search space,
                                         eter optimization. Domhan et al. proposed a method that uses
                                         parametric models to predict the learning curve [8]. Klein et al.     • Efficient sampling algorithm and pruning algorithm that
                                         constructed Bayesian neural networks to predict the expected            allows some user-customization,
                                         learning curve [9]. Li et al. employed a bandit-based algorithm
                                                                                                               • Easy-to-setup, versatile architecture that can be deployed
                                         and proposed Hyperband [10].
                                                                                                                 for tasks of various types, ranging from light-weight experi-
                                         A still another way to accelerate the optimization process is to        ments conducted via interactive interfaces to heavy-weight
                                         use distributed computing, which enables parallel processing of         distributed computations.
                                         multiple trials. Katib is built on Kubeflow, which is a computing
                                         platform for machine learning services that is based on Kuber- In this study, we will demonstrate the significance of these
                                         netes. Tune also supports parallel optimization, and uses the Ray criteria through Optuna, an open-source optimization software
                                         distributed computing platform [11].                                which is a culmination of our effort in making our definition of
                                                                                                             next-generation optimization framework come to reality.
                                         However, there are several serious problems that are being
                                         overlooked in many of these existing optimization frameworks. We will also present new design techniques and new optimiza-
                                         Firstly, all previous hyperparameter optimization frameworks to tion algorithms that we had to develop in order to meet our
                                         date require the user to statically construct the parameter-search- proposed criteria. Thanks to these new design techniques, our
                                         space for each model, and the search space can be extremely implementation outperforms many major black-box optimiza-
                                         hard to describe in these frameworks for large-scale experiments tion frameworks while being easy to use and easy to setup in
                                         that involve massive number of candidate models of different various environments. In what follows, we will elaborate each
                                         types with large parameter spaces and many conditional vari- of our proposed criteria together with our technical solutions,
                                         ables. When the parameter space is not appropriately described and present experimental results in both real world applications
                                         by the user, application of advanced optimization method can and benchmark datasets.


--- PAGE BREAK ---

Preprint – Optuna: A Next-generation Hyperparameter Optimization Framework                                                                   2

 1   import optuna                                                           2    Define-by-run API
 2   import ...
 3
 4   def objective (trial ):                                                 In this section we describe the significance of the define-by-run
 5       n layers = trial . suggest int (’ n layers ’, 1, 4)                 principle. As we will elaborate later, we are borrowing the
 6
 7        layers = []                                                        term define-by-run from a trending philosophy in deep learning
 8        for i in range( n layers ):
 9            layers . append ( trial . suggest int (’ n u n i t s l {} ’.   frameworks that allows the user to dynamically program deep
                   format (i), 1, 128))                                      networks. Following the original definition, we use the term
10
11
12
          clf = MLPClassifier ( tuple( layers ))                             define-by-run in the context of optimization framework to refer
13        mnist = fetch mldata (’MNIST original ’)                           to a design that allows the user to dynamically construct the
14        x train , x test , y train , y test = train test split (
               mnist.data , mnist . target )                                 search space. In define-by-run API, the user does not have to
15
16        clf.fit( x train , y train )                                       bear the full burden of explicitly defining everything in advance
17
18        return 1.0 − clf. score ( x test , y test )
                                                                             about the optimization strategy.
19
20   study = optuna . create study ()                           The power of define-by-run API is more easily understood with
21   study. optimize (objective , n trials =100)
                                                                actual code. Optuna formulates the hyperparameter optimization
                                                                as a process of minimizing/maximizing an objective function
Figure 1: An example code of Optuna’s define-by-run style API. that takes a set of hyperparameters as an input and returns its
This code builds a space of hyperparameters for a classifier of (validation) score. Figure 1 is an example of an objective func-
the MNIST dataset and optimizes the number of layers and the tion written in Optuna. This function dynamically constructs
number of hidden units at each layer.                           the search space of neural network architecture (the number
                                                                of layers and the number of hidden units) without relying on
                                                                externally defined static variables. Optuna refers to each process
                                                                of optimization as a study, and to each evaluation of objective
1   import hyperopt
                                                                function as a trial. In the code of Figure 1, Optuna defines an ob-
2
3
    import ...                                                  jective function (Lines 4–18), and invokes the ‘optimize API’
4   space = {                                                   that takes the objective function as an input (Line 21). Instead of
5       ’ n units l1 ’: hp. randint (’ n units l1 ’, 128) ,     hyperparameter values, an objective function in Optuna receives
6       ’l2’: hp. choice (’l2 ’, [ {
7            ’ has l2 ’: True ,                                 a living trial object, which is associated with a single trial.
 8              ’ n units l2 ’: hp. randint (’ n units l2 ’, 128) ,
 9              ’l3’: hp. choice (’l3 ’, [ {                                 Optuna gradually builds the objective function through the inter-
10                    ’ has l3 ’: True ,
11                    ’ n units l3 ’: hp. randint (’ n units l3 ’, 128) ,    action with the trial object. The search spaces are constructed
12                    ’l4’: hp. choice (’l4 ’, [ {                           dynamically by the methods of the trial object during the run-
13                          ’ has l4 ’: True ,
14                          ’ n units l4 ’: hp. randint (’ n units l4 ’,     time of the objective function. The user is asked to invoke
                                  128) ,                                     ‘suggest API’ inside the objective function in order to dynami-
15                    } , { ’ has l4 ’: False } ]) ,
16              } , { ’ has l3 ’: False } ]) ,                               cally generate the hyperparameters for each trial (Lines 5 and
17        } , { ’ has l2 ’: False } ]) ,                                     9). Upon the invocation of ‘suggest API’, a hyperparameter
18   }                                                                       is statistically sampled based on the history of previously evalu-
19
20   def objective (space ):                                                 ated trials. At Line 5, ‘suggest int’ method suggests a value
21       layers = [space [’ n units l1 ’] + 1]
22       for i in range (2, 5):                                              for ‘n layers’, the integer hyperparameter that determines the
23           space = space [’l {} ’. format (i)]                             number of layers in the Multilayer Perceptron. Using loops
24           if not space [’ has l {} ’. format (i)]:
25               break                                                       and conditional statements written in usual Python syntax, the
26           layers . append ( space [’ n u n i t s l {} ’. format (i)] +    user can easily represent a wide variety of parameter spaces.
                  1)
27                                                                           With this functionality, the user of Optuna can even express het-
28        clf = MLPClassifier ( tuple( layers ))                             erogeneous parameter space with an intuitive and simple code
29
30
31
          mnist = fetch mldata (’MNIST original ’)
          x train , x test , y train , y test = train test split (
                                                                             (Figure 3).
               mnist.data , mnist . target )
32                                                                Meanwhile, Figure 2 is an example code of Hyperopt that has
33        clf.fit( x train , y train )
34                                                                the exactly same functionality as the Optuna code in Figure 1.
35      return 1.0 − clf. score ( x test , y test )               Note that the same function written in Hyperopt (Figure 2) is
36
37  hyperopt .fmin(fn=objective , space =space , max evals =100 ,
         algo= hyperopt .tpe. suggest )
                                                                  significantly longer, more convoluted, and harder to interpret. It
                                                                  is not even obvious at first glance that the code in Figure 2 is
                                                                  in fact equivalent to the code in Figure 1! In order to write the
Figure 2: An example code of Hyperopt [1] that has the exactly same for-loop in Figure 1 using Hyperopt, the user must prepare
same functionality as the code in 1. Hyperopt is an example of the list of all the parameters in the parameter-space prior to the
define-and-run style API.                                         exploration (see line 4-18 in Figure 2). This requirement will
                                                                  lead the user to even darker nightmares when the optimization
                                                                  problem is more complicated.

Optuna is released under the MIT license (https://github. 2.1 Modular Programming
com/pfnet/optuna/), and is in production use at Preferred
                                                          A keen reader might have noticed in Figure 3 that the optimiza-
Networks for more than one year.
                                                          tion code written in Optuna is highly modular, thanks to its
                                                          define-by-run design. Compatibility with modular programming


--- PAGE BREAK ---

Preprint – Optuna: A Next-generation Hyperparameter Optimization Framework                                                                                          3

Table 1: Software frameworks for deep learning and hyperparameter optimization, sorted by their API styles: define-and-run and
define-by-run.
                                                    Deep Learning Frameworks                                 Hyperparameter Optimization Frameworks

               Define-and-Run Style          Torch (2002), Theano (2007), Caffe (2013),            SMAC (2011), Spearmint (2012), Hyperopt (2015), GPyOpt (2016),
                 (symbolic, static)        TensorFlow (2015), MXNet (2015), Keras (2015)              Vizier (2017), Katib (2018), Tune (2018), Autotune (2018)
                Define-by-Run Style        Chainer (2015), DyNet (2016), PyTorch (2016),
                                                                                                                       Optuna (2019; this work)
               (imperative, dynamic)          TensorFlow Eager (2017), Gluon (2017)


Table 2: Comparison of previous hyperparameter optimization frameworks and Optuna. There is a checkmark for lightweight if
the setup for the framework is easy and it can be easily used for lightweight purposes.

                        Framework                API Style          Pruning          Lightweight              Distributed         Dashboard         OSS
                        SMAC [3]              define-and-run            7                      3                   7                   7              3
                         GPyOpt               define-and-run            7                      3                   7                   7              3
                       Spearmint [2]          define-and-run            7                      3                   3                   7              3
                       Hyperopt [1]           define-and-run            7                      3                   3                   7              3
                       Autotune [4]           define-and-run            3                      7                   3                   3              7
                         Vizier [5]           define-and-run            3                      7                   3                   3              7
                           Katib              define-and-run            3                      7                   3                   3              3
                         Tune [7]             define-and-run            3                      7                   3                   3              3
                      Optuna (this work)       define-by-run            3                      3                   3                   3              3



 1   import sklearn                                                                        1   import chainer
 2   import ...                                                                            2   import ...
 3                                                                                         3
 4   def create rf ( trial ):                                                              4   def create model ( trial ):
 5       rf max depth = trial . suggest int (’ rf max depth ’, 2,                          5       n layers = trial . suggest int (’ n layers ’, 1, 3)
              32)                                                                          6       layers = []
 6       return RandomForestClassifier ( max depth =                                       7       for i in range( n layers ):
              rf max depth )                                                               8           n units = trial . suggest int (’ n u n i t s l {} ’. format
 7
                                                                                                              (i), 4, 128)
 8   def create mlp (trial ):                                                           9              layers . append (L. Linear (None , n units ))
 9       n layers = trial . suggest int (’ n layers ’, 1, 4)                           10              layers . append (F.relu)
10
11       layers = []                                                                   11          layers . append (L. Linear (None , 10))
12       for i in range( n layers ):                                                   12          return chainer . Sequential(* layers )
                                                                                       13
13           layers . append ( trial . suggest int (’ n u n i t s l {} ’.              14      def create optimizer (trial , model ):
                  format (i), 1, 128))                                                 15          lr = trial . suggest loguniform (’lr ’, 1e−5, 1e−1)
14                                                                                     16          optimizer = chainer . optimizers . MomentumSGD (lr=lr)
15       return MLPClassifier (tuple ( layers ))                                       17          weight decay = trial . suggest loguniform (’
16                                                                                                      weight decay ’, 1e −10, 1e−3)
17   def objective (trial ):                                                           18          optimizer . setup ( model )
18       classifier name = trial . suggest categorical (’                              19          optimizer . add hook ( chainer . optimizer . WeightDecay (
              classifier ’, [’rf ’ , ’mlp ’])                                                           weight decay ))
19                                                                                     20          return optimizer
20       if classifier name == ’rf ’:                                                  21
21           classifier obj = create rf ( trial )                                      22      def objective ( trial ):
22       else:                                                                         23          model = create model ( trial )
23           classifier obj = create mlp ( trial )                                     24          optimizer = create optimizer (trial , model)
24                                                                                     25          ...
25       ...                                                                           26
                                                                                       27      study = optuna . create study ()
                                                                                       28      study . optimize (objective , n trials =100)
Figure 3: An example code of Optuna for the construction of
a heterogeneous parameter-space. This code simultaneously Figure 4: Another example of Optuna’s objective function. This
explores the parameter spaces of both random forest and MLP. code simultaneously optimizes neural network architecture (the
                                                             create model method) and the hyperparameters for stochastic
                                                             gradient descent (the create optimizer method).
 is another important strength of the define-by-run design. Fig-
 ure 4 is another example code written in Optuna for a more
 complex scenario. This code is capable of simultaneously op- expressed in Optuna. Most notably, in this example, the methods
 timizing both the topology of a multilayer perceptron (method ‘create model’ and ‘create optimizer’ are independent of
‘create model’) and the hyperparameters of stochastic gra- one another, so that we can make changes to each one of them
 dient descent (method ‘create optimizer’). The method separately. Thus, the user can easily augment this code with
‘create model’ generates ‘n layers’ in Line 5 and uses a for other conditional variables and methods for other set of parame-
 loop to construct a neural network of depth equal to ‘n layers’. ters, and make a choice from more diverse pool of models.
The method also generates ‘n units i’ at each i-th loop, a hy-
 perparameter that determines the number of the units in the i-th 2.2 Deployment
 layer. The method ‘create optimizer’, on the other hand,
 makes suggestions for both learning rate and weight-decay pa- Indeed, the benefit of our define-by-run API means nothing if
 rameter. Again, a complex space of hyperparameters is simply we cannot easily deploy the model with the best set of hyperpa-


--- PAGE BREAK ---

Preprint – Optuna: A Next-generation Hyperparameter Optimization Framework                                                               4

 rameters found by the algorithm. The above example (Figure 4)         existing frameworks do not provide efficient pruning strategies.
 might make it seem as if the user has to write a different version    In this section we will provide our design for both sampling and
 of the objective function that does not invoke ‘trial.suggest’        pruning.
 in order to deploy the best configuration. Luckily, this is not
 a concern. For deployment purpose, Optuna features a sepa-            3.1   Sampling Methods on Dynamically Constructed
 rate class called ‘FixedTrial’ that can be passed to objective              Parameter Space
 functions. The ‘FixedTrial’ object has practically the same
 set of functionalities as the trial class, except that it will only   There are generally two types of sampling method: relational
 suggest the user defined set of the hyperparameters when passed       sampling that exploits the correlations among the parameters
 to the objective functions. Once a parameter-set of interest is       and independent sampling that samples each parameter inde-
 found (e.g., the best ones), the user simply has to construct a       pendently. The independent sampling is not necessarily a naive
‘FixedTrial’ object with the parameter set.                            option, because some sampling algorithms like TPE [6] are
                                                                       known to perform well even without using the parameter correla-
2.3   Historical Remarks                                               tions, and the cost effectiveness for both relational and indepen-
                                                                       dent sampling depends on environment and task. Our Optuna
Historically, the term define-by-run was coined by the develop-        features both, and it can handle various independent sampling
ers of deep learning frameworks. In the beginning, most deep           methods including TPE as well as relational sampling methods
learning frameworks like Theano and Torch used to be declar-           like CMA-ES. However, some words of caution are in order
ative, and constructed the networks in their domain specific           for the implementation of relational sampling in define-by-run
languages (DSL). These frameworks are called define-and-run            framework.
frameworks because they do not allow the user to alter the ma-
nipulation of intermediate variables once the network is defined.      Relational sampling in define-by-run frameworks
In define-and-run frameworks, computation is conducted in two
phases: (1) construction phase and (2) evaluation phase. In a          One valid claim about the advantage of the old define-and-run
way, contemporary optimization methods like Hyperopt are built         optimization design is that the program is given the knowledge
on the philosophy similar to define-and-run, because there are         of the concurrence relations among the hyperparamters from the
two phases in their optimization: (1) construction of the search       beginning of the optimization process. Implementing of opti-
space and (3) exploration in the search space.                         mization methods that takes the concurrence relations among
                                                                       the parameters into account is a nontrivial challenge when the
Because of their difficulty of programming, the define-and-run
                                                                       search spaces are dynamically constructed. To overcome this
style deep learning frameworks are quickly being replaced by
                                                                       challenge, Optuna features an ability to identify trial results that
define-by-run style deep learning frameworks like Chainer [13],
                                                                       are informative about the concurrence relations. This way, the
DyNet [14], PyTorch [15], eager-mode TensorFlow [16], and
                                                                       framework can identify the underlying concurrence relations
Gluon. In the define-by-run style DL framework, there are no
                                                                       after some number of independent samplings, and use the in-
two separate phases for the construction of the network and the
                                                                       ferred concurrence relation to conduct user-selected relational
computation on the network. Instead, the user is allowed to
                                                                       sampling algorithms like CMA-ES [17] and GP-BO [18]. Being
directly program how each variables are to be manipulated in
                                                                       an open source software, Optuna also allows the user to use
the network. What we propose in this article is an analogue of
                                                                       his/her own customized sampling procedure.
the define-by-run DL framework for hyperparameter optimiza-
tion, in which the framework asks the user to directly program
the parameter search-space (See Table 1) . Armed with the 3.2 Efficient Pruning Algorithm
architecture built on the define-by-run principle, our Optuna can
                                                                     Pruning algorithm is essential in ensuring the ”cost” part of the
express highly sophisticated search space at ease.
                                                                     cost-effectiveness. Pruning mechanism in general works in two
                                                                     phases. It (1) periodically monitors the intermediate objective
3 Efficient Sampling and Pruning Mechanism                           values, and (2) terminates the trial that does not meet the prede-
                                                                     fined condition. In Optuna, ‘report API’ is responsible for the
In general, the cost-effectiveness of hyperparameter optimiza- monitoring functionality, and ‘should prune API’ is respon-
tion framework is determined by the efficiency of (1) searching sible for the premature termination of the unpromising trials
strategy that determines the set of parameters that shall be inves- (see Figure 5). The background algorithm of ‘should prune’
tigated, and (2) performance estimation strategy that estimates method is implemented by the family of pruner classes. Op-
the value of currently investigated parameters from learning tuna features a variant of Asynchronous Successive Halving
curves and determines the set of parameters that shall be dis- algorithm [19] , a recently developed state of the art method
carded. As we will experimentally show later, the efficiency of that scales linearly with the number of workers in distributed
both searching strategy and performance estimation strategy is environment.
necessary for cost-effective optimization method.
                                                                     Asynchronous Successive Halving(ASHA) is an extension of
The strategy for the termination of unpromising trials is often Successive Halving [20] in which each worker is allowed to
referred to as pruning in many literatures, and it is also well asynchronously execute aggressive early stopping based on pro-
known as automated early stopping [5] [7]. We, however, refer visional ranking of trials. The most prominent advantage of
to this functionality as pruning in order to distinguish it from the asynchronous pruning is that it is particularly well suited for
early stopping regularization in machine learning that exists as applications in distributional environment; because each worker
a countermeasure against overfitting. As shown in table 2, many does not have to wait for the results from other workers at each


--- PAGE BREAK ---

Preprint – Optuna: A Next-generation Hyperparameter Optimization Framework                                                           5

 1   import ...
 2
 3   def objective (trial ):
 4       ...
 5
 6       lr = trial. suggest loguniform (’lr ’, 1e−5, 1e−1)
 7       clf = sklearn . linear model . SGDClassifier (
              learning rate =lr)
 8       for step in range (100) :
 9           clf. partial fit ( x train , y train , classes )
10
11           # Report intermediate objective value .
12           intermediate value = clf. score ( x val , y val )
13           trial. report ( intermediate value , step=step)
14
15           # Handle pruning based on the intermediate value
                  .
16           if trial . should prune (step):
17               raise TrialPruned ()
18
19       return 1.0 − clf. score ( x val , y val )
                                                               Figure 6: Overview of Optuna’s system design. Each worker
20
21  study = optuna . create study ()
                                                               executes one instance of an objective function in each study.
22  study. optimize ( objective )                              The Objective function runs its trial using Optuna APIs. When
                                                               the API is invoked, the objective function accesses the shared
Figure 5: An example of implementation of a pruning algorithm storage and obtains the information of the past studies from the
with Optuna. An intermediate value is reported at each step of storage when necessary. Each worker runs the objective function
iterative training. The Pruner class stops unpromising trials independently and shares the progress of the current study via
based on the history of reported values.                       the storage.



  Algorithm 1: Pruning algorithm based on Successive Halv-
   ing                                                               4   Scalable and versatile System that is Easy to
    Input: target trial trial, current step step, minimum                setup
           resource r, reduction factor η, minimum
           early-stopping rate s.                                    Our last criterion for the next generation optimization software
    Output: true if the trial should be pruned, false otherwise.     is a scalable system that can handle a wide variety of tasks,
 1 rung ← max(0, log η(bstep/rc) − s)                                ranging from a heavy experiment that requires a massive number
                s+rung
 2 if step , rη         then                                         of workers to a trial-level, light-weight computation conducted
 3     return false                                                  through interactive interfaces like Jupyter Notebook. The figure
 4 end                                                               6 illustrates how the database(storage) is incorporated into the
 5 value ←get trial intermediate value(trial, step)                  system of Optuna; the trial objects shares the evaluations history
 6 values ←get all trials intermediate values(step)                  of objective functions via storage. Optuna features a mechanism
 7 top k values ←top k(values, b|values|/ηc)                         that allows the user to change the storage backend to meet his/her
 8 if top k values = ∅ then                                          need.
 9     top k values ←top k(values, 1)
10 end
                                                                     For example, when the user wants to run experiment with Jupyter
11 return value < top k values
                                                                     Notebook in a local machine, the user may want to avoid spend-
                                                                     ing effort in accessing a multi-tenant system deployed by some
                                                                     organization or in deploying a database on his/her own. When
                                                                     there is no specification given, Optuna automatically uses its
                                                                     built-in in-memory data-structure as the storage back-end. From
round of the pruning, the parallel computation can process mul-      general user’s perspective, that the framework can be easily used
tiple trials simultaneously without delay.                           for lightweight purposes is one of the most essential strengths
                                                                     of Optuna, and it is a particularly important part of our criteria
Algorithm 1 is the actual pruning algorithm implemented in
                                                                     for next-generation optimization frameworks. This lightweight
Optuna. Inputs to the algorithm include the trial that is subject
                                                                     purpose compatibility is also featured by select few frameworks
to pruning, number of steps, reducing factor, minimum resource
                                                                     like Hyperopt and GPyOt as well. The user of Optuna can also
to be used before the pruning, and minimum early stopping rate.
                                                                     conduct more involved analysis by exporting the results in the
Algorithm begins by computing the current rung for the trial,
                                                                     pandas [21] dataframe, which is highly compatible with interac-
which is the number of times the trial has survived the pruning.
                                                                     tive analysis frameworks like Jupyter Notebooks [22]. Optuna
The trial is allowed to enter the next round of the competition
                                                                     also provides web-dashboard for visualization and analysis of
if its provisional ranking is within top 1/η. If the number of
                                                                     studies in real time (see Figure 8).
trials with the same rung is less than η, the best trial among the
trials with the same rung becomes promoted. In order to avoid        Meanwhile, when the user wants to conduct distributed com-
having to record massive number of checkpointed configura-           putation, the user of Optuna can deploy relational database as
tions(snapshots), our implementation does not allow repechage.       the backend. The user of Optuna can also use SQLite database
As experimentally verify in the next section, our modified imple-    as well. The figure 7b is an example code that deploys SQLite
mentation of Successive Halving scales linearly with the number      database. This code conducts distributed computation by simply
of workers without any problem. We will present the details of       executing run.py multiple times with the same study identifier
our optimization performance in Section 5.2.                         and the same storage URL.


--- PAGE BREAK ---

Preprint – Optuna: A Next-generation Hyperparameter Optimization Framework                                                       6

 1   import ...
 2
 3   def objective (trial ):
 4       ...
 5       return objective value
 6
 7   study name = sys.argv [1]
 8   storage = sys.argv [2]
 9
10   study = optuna . Study ( study name , storage )
11   study. optimize ( objective )


                       (a) Python code: run.py


 1   # Setup: the shared storage URL and study identifier .
 2   STORAGE URL = ' sqlite :/// example .db '
 3   STUDY ID =$( optuna create−study −−storage $ STORAGE URL )
 4
 5   # Run the script from multiple processes and/or nodes .
                                                                   Figure 9: Result of comparing TPE+CMA-ES against other
 6   # Their execution can be asynchronous .                       existing methods in terms of best attained objective value. Each
 7   python run.py $ STUDY ID $ STORAGE URL &                      algorithm was applied to each study 30 times, and Paired Mann-
     python run.py $ STUDY ID $ STORAGE URL &
 8
 9   python run.py $ STUDY ID $ STORAGE URL &
                                                                   Whitney U test with α = 0.0005 was used to determine whether
10   ...                                                           TPE+CMA-ES outperforms each rival.

                               (b) Shell
Figure 7: Distributed optimization in Optuna. Figure (a) is the
optimization script executed by one worker. Figure (b) is an
example shell for the optimization with multiple workers in a
distributed environment.
                                                                  the purpose, but also comes with multiple built-in optimization
                                                                  algorithms including the mixture of independent and relational
                                                                  sampling, which is not featured in currently existing frameworks.
                                                                  For example, Optuna can use the mixture of TPE and CMA-ES.
                                                                  We compared the optimization performance of the TPE+CMA-
                                                                  ES against those of other sampling algorithms on a collection
                                                                  of tests for black-box optimization [23, 24], which contains 56
                                                                  test cases. We implemented four adversaries to compare against
                                                                  TPE+CMA-ES: random search as a baseline method, Hyper-
                                                                  opt [1] as a TPE-based method, SMAC3 [3] as a random-forest
                                                                  based method, and GPyOpt as a Gaussian Process based method.
                                                                  For TPE+CMA-ES, we used TPE for the first 40 steps and used
Figure 8: Optuna dashboard. This example shows the online CMA-ES for the rest. For the evaluation metric, we used the
transition of objective values, the parallel coordinates plot of best-attained objective value found in 80 trials. Following the
sampled parameters, the learning curves, and the tabular descrip- work of Dewancker et al. [24], we repeated each study 30 times
tions of investigated trials.                                     for each algorithm and applied Paired Mann-Whitney U test
                                                                  with α = 0.0005 to the results in order to statistically compare
                                                                  TPE+CMA-ES’s performance against the rival algorithms.
Optuna’s new design thus significantly reduces the effort re-      The results are shown in Figure 9. TPE+CMA-ES finds sta-
quired for storage deployment. This new design can be easily in-   tistically worse solution than random search in only 1/56 test
corporated into a container-orchestration system like Kubernetes   cases, performs worse than Hyperopt in 1/56 cases, and per-
as well. As we verify in the experiment section, the distributed   forms worse than SMAC3 in 3/56 cases. Meanwhile, GPyOpt
computations conducted with our flexible system-design scales      performed better than TPE+CMA-ES in 34/56 cases in terms of
linearly with the number of workers. Optuna is also an open        the best-attained loss value. At the same time, TPE+CMA-ES
source software that can be installed to user’s system with one    takes an order-of-magnitude less times per trial than GPyOpt.
command.
                                                                 Figure 10 shows the average time spent for each test case.
                                                                 TPE+CMA-ES, Hyperopt, SMAC3, and random search finished
5 Experimental Evaluation                                        one study within few seconds even for the test case with more
                                                                 than ten design variables. On the other hand, GPyOpt required
We demonstrate the efficiency of the new design-framework twenty times longer duration to complete a study. We see that the
through three sets of experiments.                               mixture of TPE and CMA-ES is a cost-effective choice among
                                                                 current lines of advanced optimization algorithms. If the time of
5.1 Performance Evaluation Using a Collection of Tests           evaluation is a bottleneck, the user may use Gaussian Process
                                                                 based method as a sampling algorithm. We plan in near future
As described in the previous section, Optuna not only allows the to also develop an interface on which the user of Optuna can
user to use his/her own customized sampling procedure that suits easily deploy external optimization software as well.


--- PAGE BREAK ---

Preprint – Optuna: A Next-generation Hyperparameter Optimization Framework                                                               7

                                                                       workers and the number of trials. This is especially the case
                                                                       for a SMBO [3] such as TPE, where the algorithm is designed
                                                                       to sequentially evaluate each trial. The result illustrated in Fig-
                                                                       ure 11c resolves this concern. Note that the optimization scores
                                                                       per the number of trials (i.e., parallelization efficiency) barely
                                                                       changes with the number of workers. This shows that the perfor-
                                                                       mance is linearly scaling with the number of trials, and hence
                                                                       with the number of workers. Figure 12 illustrates the result of
                                                                       optimization that uses both parallel computation and pruning.
                                                                       The result suggests that our optimization scales linearly with
                                                                       the number of workers even when implemented with a pruning
                                                                       algorithm.
Figure 10: Computational time spent by different frameworks
for each test case.                                                    6   Real World Applications
                                                                       Optuna is already in production use, and it has been success-
5.2   Performance Evaluation of Pruning                                fully applied to a number of real world applications. Optuna is
                                                                       also being actively used by third parties for various purposes,
We evaluated the performance gain from the pruning procedure           including projects based on TensorFlow and PyTorch. Some
in the Optuna-implemented optimization of Alex Krizhevsky’s            projects use Optuna as a part of pipeline for machine-learning
neural network (AlexNet) [25] on the Street View House Num-            framework (e.g., redshells2 , pyannote-pipeline3 ). In this section,
bers (SVHN) dataset [26]. We tested our pruning system to-             we present the examples of Optuna’s applications in the projects
gether with random search and TPE. Following the experiment            at Preferred Networks.
in [10], we used a subnetwork of AlexNet (hereinafter called
simplified AlexNet), which consists of three convolutional layers Open Images Object Detection Track 2018. Optuna was a
and a fully-connected layer and involves 8 hyperparameters.         key player in the development of Preferred Networks’ Faster-
                                                                    RCNN models for Google AI Open Images Object Detection
For each experiment, we executed a study with one NVIDIA Track 2018 on Kaggle 4 , whose dataset is at present the largest
Tesla P100 card, and terminated each study 4 hours into the in the field of object detection [27]. Our final model, PFDet [28],
experiment. We repeated each study 40 times. With pruning, won the 2nd place in the competition.
both TPE and random search was able to conduct a greater
number of trials within the same time limit. On average, TPE As a versatile next generation optimization software, Optuna
and random search without pruning completed 35.8 and 36.0 can be used in applications outside the field of machine learning
trials per study, respectively. On the other hand, TPE with as well. Followings are applications of Optuna for non-machine
pruning explored 1278.6 trials on average per study, of which learning tasks.
1271.5 were pruned during the process. Random search with
pruning explored 1119.3 trials with 1111.3 pruned trials.           High Performance Linpack for TOP500. The Linpack bench-
Figure 11a shows the transition of the average test errors. The mark is a task whose purpose is to measure the floating point
result clearly suggests that pruning can significantly accelerate computation power of a system in which the system is asked
the optimization for both TPE and random search. Our imple- to solve a dense matrix LU factorization. The performance on
mentation of ASHA significantly outperforms Median pruning, this task is used as a measure of sheer computing power of a
a pruning method featured in Vizier. This result also suggests system  5
                                                                            and is used to rank the supercomputers in the TOP500
that sampling algorithm alone is not sufficient for cost-effective list . High Performance Linpack (HPL) is one of the implemen-
optimization. The bottleneck of sampling algorithm is the com- tations for Linpack. HPL involves many hyperparameters, and
putational cost required for each trial, and pruning algorithm is the performance result of any system heavily relies on them.
necessary for fast optimization.                                    We used Optuna to optimize these hyperparameters in the eval-
                                                                    uation of the maximum performance of MN-1b, an in-house
                                                                    supercomputer owned by Preferred Networks.
5.3 Performance Evaluation of Distributed Optimization
                                                                    RocksDB. RocksDB [29] is a persistent key-value store for fast
We also evaluated the scalability of Optuna’s distributed opti-
                                                                    storage that has over hundred user-customizable parameters. As
mization. Based on the same experimental setup used in Sec-
                                                                    described by the developers in the official website, ”configuring
tion 5.2, we recorded the transition of the best scores obtained
                                                                    RocksDB optimally is not trivial”, and even the ”RocksDB de-
by TPE with 1, 2, 4, and 8 workers in a distributed environment.
                                                                    velopers don’t fully understand the effect of each configuration
Figure 11b shows the relationship between optimization score
                                                                    change”6 . For this experiment, we prepared a set of 500,000 files
and execution time. We can see that the convergence speed
increases with the number of workers.                             2
                                                                    https://github.com/m3dev/redshells
                                                                   3
In the interpretation of this experimental results, however, we 4 https://github.com/pyannote/pyannote-pipeline
                                                                     https://www.kaggle.com/c/google-ai-open-images-object-detection-track
have to give a consideration to the fact that the relationship be- 5 https://www.top500.org/
tween the number of workers and the efficiency of optimization 6 https://github.com/facebook/rocksdb/wiki/RocksDB-Tuning-Guide#
is not as intuitive as the relationship between the number of          final-thoughts


--- PAGE BREAK ---

    Preprint – Optuna: A Next-generation Hyperparameter Optimization Framework                                                               8




                          (a)                                           (b)                                             (c)
    Figure 11: The transition of average test errors of simplified AlexNet for SVHN dataset. Figure (a) illustrates the effect of pruning
    mechanisms on TPE and random search. Figure (b) illustrates the effect of the number of workers on the performance. Figure (c)
    plots the test errors against the number of trials for different number of workers. Note that the number of workers has no effect on
    the relation between the number of executed trials and the test error. The result also shows the superiority of ASHA pruning over
    median pruning.


                                                                            Buck Bunny”8 . Optuna was able to find a parameter-set whose
                                                                            performance is on par with the second best parameter-set among
                                                                            the presets provided by the developers.


                                                                            7    Conclusions
                                                                            The efficacy of Optuna strongly supports our claim that our new
                                                                            design criteria for next generation optimization frameworks are
                                                                            worth adopting in the development of future frameworks. The
                                                                            define-by-run principle enables the user to dynamically construct
                                                                            the search space in the way that has never been possible with
    Figure 12: Distributed hyperparameter optimization process for          previous hyperparameter tuning frameworks. Combination of
    the minimization of average test errors of simplified AlexNet for       efficient searching and pruning algorithm greatly improves the
    SVHN dataset. The optimization was done with ASHA pruning.              cost effectiveness of optimization. Finally, scalable and versatile
                                                                            design allows users of various types to deploy the frameworks
                                                                            for a wide variety of purposes. As an open source software,
                                                                            Optuna itself can also evolve even further as a next generation
    of size 10KB each, and used Optuna to look for parameter-set
                                                                            software by interacting with open source community. It is our
    that minimizes the computation time required for applying a
                                                                            strong hope that the set of design techniques we developed for
    certain set of operations(store, search, delete) to this file set. Out
                                                                            Optuna will serve as a basis of other next generation optimiza-
    of over hundred customizable parameters, we used Optuna to
                                                                            tion frameworks to be developed in the future.
    explore the space of 34 parameters. With the default parameter
    setting, RocksDB takes 372seconds on HDD to apply the set of
    operation to the file set. With pruning, Optuna was able to find Acknowledgement. The authors thank R. Calland, S. Tokui,
    a parameter-set that reduces the computation time to 30 seconds. H. Maruyama, K. Fukuda, K. Nakago, M. Yoshikawa, M. Abe, H. Ima-
    Within the same 4 hours, the algorithm with pruning explores mura, and Y. Kitamura for valuable feedback and suggestion.
    937 sets of parameters while the algorithm without pruning only
    explores 39. When we disable the time-out option for the eval-
    uation process, the algorithm without pruning explores only 2 References
    trials. This experiment again verifies the crucial role of pruning. [1] James Bergstra, Brent Komer, Chris Eliasmith, Dan
                                                                               Yamins, and David D Cox. Hyperopt: a Python library for
    Encoder Parameters for FFmpeg. FFmpeg7 is a multimedia                     model selection and hyperparameter optimization. Compu-
    framework that is widely used in the world for decoding, en-               tational Science & Discovery, 8(1):14008, 2015.
    coding and streaming of movies and audio dataset. FFmpeg
    has numerous customizable parameters for encoding. However,            [2] Jasper Snoek, Hugo Larochelle, and Ryan P Adams. Prac-
    finding of good encoding parameter-set for FFmpeg is a non-                tical bayesian optimization of machine learning algorithms.
    trivial task, as it requires expert knowledge of codec. We used            In NIPS, pages 2951–2959, 2012.
    Optuna to seek the encoding parameter-set that minimizes the           [3] Frank Hutter, Holger H. Hoos, and Kevin Leyton-Brown.
    reconstruction error for the Blender Open Movie Project’s ”Big             Sequential model-based optimization for general algorithm
7                                                                       8
    https://www.ffmpeg.org/                                                 Blender Foundation — www.blender.org


--- PAGE BREAK ---

Preprint – Optuna: A Next-generation Hyperparameter Optimization Framework                                                        9

     configuration. In LION, pages 507–523, 2011. ISBN 978- [16] Martı́n Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen,
     3-642-25565-6.                                                    Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghe-
 [4] Patrick Koch, Oleg Golovidov, Steven Gardner, Brett Wu-           mawat, Geoffrey Irving, Michael Isard, Manjunath Kudlur,
     jek, Joshua Griffin, and Yan Xu. Autotune: A derivative-          Josh  Levenberg, Rajat Monga, Sherry Moore, Derek G.
     free optimization framework for hyperparameter tuning.            Murray, Benoit Steiner, Paul Tucker, Vijay Vasudevan,
     In KDD, pages 443–452, 2018. ISBN 978-1-4503-5552-0.              Pete Warden, Martin Wicke, Yuan Yu, and Xiaoqiang
                                                                       Zheng. TensorFlow: A system for large-scale machine
 [5] Daniel Golovin, Benjamin Solnik, Subhodeep Moitra,                learning. In OSDI, pages 265–283, 2016.
     Greg Kochanski, John Karro, and D Sculley. Google
     Vizier: A service for black-box optimization. In KDD,        [17] Nikolaus Hansen and Andreas Ostermeier. Completely
     pages 1487–1495, 2017. ISBN 978-1-4503-4887-4.                    derandomized self-adaptation in evolution strategies. Evo-
                                                                       lutionary Computation, 9(2):159–195, 2001.
 [6] James Bergstra, Rémi Bardenet, Yoshua Bengio, and
     Balázs Kégl. Algorithms for hyper-parameter optimization. [18] Bobak Shahriari, Kevin Swersky, Ziyu Wang, Ryan P
     In NIPS, pages 2546–2554, 2011.                                   Adams, and Nando De Freitas. Taking the human out of
                                                                       the loop: A review of bayesian optimization. Proceedings
 [7] Richard Liaw, Eric Liang, Robert Nishihara, Philipp               of the IEEE, 104(1):148–175, 2016.
     Moritz, Joseph E. Gonzalez, and Ion Stoica. Tune: A
     research platform for distributed model selection and train- [19] Liam Li, Kevin Jamieson, Afshin Rostamizadeh, Ekate-
     ing. In ICML Workshop on AutoML, 2018.                            rina Gonina, Moritz Hardt, Benjamin Recht, and Ameet
                                                                       Talwalkar. Massively parallel hyperparameter tuning. In
 [8] Tobias Domhan, Jost Tobias Springenberg, and Frank Hut-           NeurIPS Workshop on Machine Learning Systems, 2018.
     ter. Speeding up automatic hyperparameter optimization of
     deep neural networks by extrapolation of learning curves.    [20] Kevin Jamieson and Ameet Talwalkar. Non-stochastic best
     In IJCAI, pages 3460–3468, 2015.                                  arm identification and hyperparameter optimization. In
                                                                       Artificial Intelligence and Statistics, pages 240–248, 2016.
 [9] Aaron Klein, Stefan Falkner, Jost Tobias Springenberg, and
     Frank Hutter. Learning curve prediction with Bayesian [21] Wes McKinney. Pandas: a foundational python library for
     neural networks. In ICLR, 2017.                                   data analysis and statistics. In SC Workshop on Python for
                                                                       High Performance and Scientific Computing, 2011.
[10] Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Ros-
     tamizadeh, and Ameet Talwalkar. Hyperband: A novel [22] Thomas Kluyver, Benjamin Ragan-Kelley, Fernando Pérez,
     bandit-based approach to hyperparameter optimization.             Brian Granger, Matthias Bussonnier, Jonathan Frederic,
     Journal of Machine Learning Research, 18(185):1–52,               Kyle Kelley, Jessica Hamrick, Jason Grout, Sylvain Cor-
     2018.                                                             lay, Paul Ivanov, Damián Avila, Safia Abdalla, and Carol
                                                                       Willing. Jupyter notebooks – a publishing format for re-
[11] Philipp Moritz, Robert Nishihara, Stephanie Wang, Alexey          producible computational workflows. In F. Loizides and
     Tumanov, Richard Liaw, Eric Liang, William Paul,                  B. Schmidt, editors, Positioning and Power in Academic
     Michael I. Jordan, and Ion Stoica.          Ray: A dis-           Publishing: Players, Agents and Agendas, pages 87 – 90.
     tributed framework for emerging AI applications. CoRR,            IOS Press, 2016.
     abs/1712.05889, 2017. URL http://arxiv.org/abs/
     1712.05889.                                                  [23] Michael McCourt. Benchmark suite of test functions
                                                                       suitable for evaluating black-box optimization strategies.
[12] Frank Hutter, Lars Kotthoff, and Joaquin Vanschoren,              https://github.com/sigopt/evalset, 2016.
     editors. Automatic Machine Learning: Methods, Sys-
     tems, Challenges. Springer, 2018. In press, available [24] Ian Dewancker, Michael McCourt, Scott Clark, Patrick
     at http://automl.org/book.                                        Hayes, Alexandra Johnson, and George Ke. A strategy for
                                                                       ranking optimization methods using multiple criteria. In
[13] Seiya Tokui, Kenta Oono, Shohei Hido, and Justin Clayton.         ICML Workshop on AutoML, pages 11–20, 2016.
     Chainer: a next-generation open source framework for
     deep learning. In NIPS Workshop on Machine Learning [25] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton.
     Systems, 2015.                                                    ImageNet classification with deep convolutional neural
                                                                       networks. In NIPS, pages 1097–1105, 2012.
[14] Graham Neubig, Chris Dyer, Yoav Goldberg, Austin
     Matthews, Waleed Ammar, Antonios Anastasopoulos, [26] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bis-
     Miguel Ballesteros, David Chiang, Daniel Clothiaux,               sacco, Bo Wu, and Andrew Y Ng. Reading digits in nat-
     Trevor Cohn, Kevin Duh, Manaal Faruqui, Cynthia Gan,              ural images with unsupervised feature learning. In NIPS
     Dan Garrette, Yangfeng Ji, Lingpeng Kong, Adhiguna Kun-           Workshop    on Deep Learning and Unsupervised Feature
     coro, Gaurav Kumar, Chaitanya Malaviya, Paul Michel,              Learning,   2011.
     Yusuke Oda, Matthew Richardson, Naomi Saphra, Swabha [27] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper R. R.
     Swayamdipta, and Pengcheng Yin. Dynet: The dynamic                Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali,
     neural network toolkit. CoRR, abs/1701.03980, 2017.               Stefan Popov, Matteo Malloci, Tom Duerig, and Vittorio
[15] Adam Paszke, Sam Gross, Soumith Chintala, Gregory                 Ferrari. The open images dataset V4: unified image classi-
     Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Al-              fication, object detection, and visual relationship detection
     ban Desmaison, Luca Antiga, and Adam Lerer. Automatic             at scale. CoRR, abs/1811.00982, 2018.
     differentiation in PyTorch. In NIPS Autodiff Workshop, [28] Takuya Akiba, Tommi Kerola, Yusuke Niitani, Toru
     2017.                                                       Ogawa, Shotaro Sano, and Shuji Suzuki. PFDet: 2nd


--- PAGE BREAK ---

Preprint – Optuna: A Next-generation Hyperparameter Optimization Framework   10

     place solution to open images challenge 2018 object detec-
     tion track. In ECCV Workshop on Open Images Challenge,
     2018.
[29] Siying Dong, Mark Callaghan, Leonidas Galanis, Dhruba
     Borthakur, Tony Savor, and Michael Strum. Optimizing
     space amplification in rocksdb. In CIDR, 2017.


--- PAGE BREAK ---


