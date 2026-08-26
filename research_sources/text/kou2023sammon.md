# Optimal {TSP} tour length estimation using {S}ammon maps

> Citation key: `kou2023sammon`  
> DOI: 10.1007/s11590-022-01937-y  
> Download URL: https://api.drum.lib.umd.edu/server/api/core/bitstreams/3aa339e3-5e9f-46e4-9890-3485a9ad6f4c/content  
> SHA-256: `81196d8930d6dd8f4c098ed23feede4861b40de29c039c84cbb5796f08a6b3e5`  

---

                                        ABSTRACT



Title of Dissertation:     Statistical and Machine Learning Approaches for Estimating
                           Optimal Solution Values in Combinatorial Optimization:
                           Applications to the Traveling Salesman
                           and Vehicle Routing Problems

                           Shuhan Kou
                           Doctor of Philosophy, 2025

Dissertation Directed by: Professor Bruce L. Golden
                           Department of Decision, Operations
                           & Information Technologies
                           Robert H. Smith School of Business


      Solving combinatorial optimization problems such as the Traveling Salesman Problem

(TSP) and Vehicle Routing Problem (VRP) to optimality is computationally challenging. This

motivates the development of efficient methods to estimate optimal solution values for combi-

natorial optimization problems without solving them exactly. Such estimation approaches offer

several benefits. First, the estimated tour lengths can serve as benchmarks to evaluate heuristic

algorithms, ensuring solution quality on new instances. Additionally, logistics companies can

leverage these estimates in operations to assess potential savings from split deliveries, helping

them determine whether to implement the classic VRP scenario or adopt the Split Delivery Vehi-

cle Routing Problem (SDVRP) senario.

      This dissertation introduces and rigorously evaluates statistical and machine learning mod-

els designed to estimate the optimal tour lengths for the TSP, VRP, and SDVRP with high accu-


--- PAGE BREAK ---

racy. By integrating problem-specific insights with data-driven approaches, we construct predic-

tive models that consistently estimate the optimal solution values within a few percent of the true

optimum across diverse problem instances.

      A key contribution of this research is a comprehensive comparative analysis of multiple pre-

dictive methodologies, including linear regression, random forest regression, multilayer percep-

tron (MLP) neural networks, and the recently introduced Kolmogorov–Arnold Network (KAN).

Our systematic evaluation demonstrates the strengths and weaknesses of each approach. We offer

practical insights into the trade-offs among accuracy, interpretability, and generalizability across

different modeling paradigms. Linear regression, while highly interpretable and computationally

efficient, may lack the flexibility to capture more complex relationships. Conversely, neural net-

work models, particularly KAN, provide a balance of interpretability and accuracy but require

more sophisticated modeling and training processes. These insights serve as valuable guidance

for practitioners and researchers, facilitating informed decisions about appropriate methods for

estimating optimal solution values in logistics planning and research.

      Additionally, we present novel high-accuracy estimation models tailored specifically for the

TSP, VRP, and SDVRP. These models leverage innovative feature extraction techniques, such as

moment statistics derived from random heuristic solutions, to effectively capture the underlying

structure of the problems. To further explore the solution space of complex routing problems, we

develop a modified Clarke & Wright algorithm with a split deliveries heuristic for the SDVRP.

Additionally, for the VRP, we provide an example illustrating how estimation models can be

applied to improve the heuristic algorithm for the problem. Extensive empirical experiments

highlight the practical utility of our models in accurately estimating optimal values.


--- PAGE BREAK ---

Statistical and Machine Learning Approaches for Estimating Optimal Solution
Values in Combinatorial Optimization: Applications to the Traveling Salesman
                        and Vehicle Routing Problems


                                             by

                                       Shuhan Kou



             Dissertation submitted to the Faculty of the Graduate School of the
                 University of Maryland, College Park in partial fulfillment
                            of the requirements for the degree of
                                    Doctor of Philosophy
                                            2025




Advisory Committee:
       Professor Bruce L. Golden, Chair/Advisor
       Professor Luca Bertazzi
       Professor Zhi-Long Chen
       Professor Vince Lyzinski
       Professor Paul Schonfeld


--- PAGE BREAK ---

© Copyright by
 Shuhan Kou
    2025


--- PAGE BREAK ---

Dedication

To my family.




     ii


--- PAGE BREAK ---

                                     Acknowledgments


      I would like to express my deepest gratitude to everyone who made this thesis possible and

contributed to an unforgettable graduate experience that I will cherish forever.

      First and foremost, I am sincerely grateful to my advisor, Professor Bruce Golden. I am

thankful for the invaluable opportunity to engage in challenging and fascinating projects under his

guidance over the past few years. His constant support, brilliant ideas, and thoughtful mentorship

have been instrumental in shaping both my research and professional growth. Beyond academics,

his kindness and encouragement during difficult times have meant a great deal to me. I will miss

our weekly meetings and his witty jokes, and I will do my best to imitate his sense of humor. It

has truly been an honor and a privilege to learn from him.

      I also extend my heartfelt thanks to my collaborators, Dr. Luca Bertazzi and Dr. Stefan

Poikonen, whose brilliant ideas and insightful feedback have greatly enriched this dissertation.

A special thanks to Dr. Luca Bertazzi for serving on my thesis committee. Additionally, I am

sincerely grateful to Professor Zhi-Long Chen, Professor Vince Lyzinski, and Professor Paul

Schonfeld for serving on my thesis committee and generously dedicating their time to reviewing

my work.

      I would also like to express my appreciation to my professors at the Mathematics De-

partment of the University of Michigan, whose exceptional teaching and guidance thoroughly

prepared me for my Ph.D. journey. In particular, I am deeply thankful to Professor Karen Smith,


                                                iii


--- PAGE BREAK ---

whose passion for teaching and research has inspired me to pursue a meaningful career. I also

want to extend my heartfelt thanks to my childhood math teacher, Xiaoling Sun, who recognized

my passion early on and provided invaluable mentorship and flexibility, shaping the path that led

me here. Without all of your support, I would not be the person I am today.

      I am incredibly fortunate to have had the friendship and support of my colleagues and

friends at the University of Maryland, including Dr. Yang Trista Cao, Yu Hou, Jiajing Guan,

Jiaxin Yuan, Haozhe An, Ruijie Zheng, Xuliang Dong, Yuancheng Xu, Yuan Gao, Yue Feng,

Xiaoyu Liu, Mengting Cao, Jiaqi Wang, Hugo Mainguy, and Yang Hong. Their encouragement

has been invaluable throughout my studies. A special thanks to Trista: I will forever cherish our

five years of living together and taking care of Bobo. I hope our friendship lasts until our black

hair turns gray.

      I am also grateful to the senior Ph.D. graduates at the University of Maryland, including

Dr. Rui Zhang, Dr. Eric Oden, and Dr. Yuchen Luo, who provided invaluable guidance when I

faced uncertainty about research and my future career. Their experiences and advice helped me

avoid many detours along the way.

      I sincerely thank Dr. Anand Subramanian for sharing his research on the near-optimal

SDVRP solver, the SplitILS solver, and for connecting me with Dr. Marcos Melo Silva. I am

also grateful to Dr. Marcos Melo Silva for providing access to the solver and for his detailed

guidance on its usage. Their support has been invaluable to my research.

      A special thanks to the dedicated staff of the Applied Mathematics & Statistics, and Sci-

entific Computation (AMSC) and Mathematics graduate programs at UMD. I am particularly

grateful to Jessica Sadler for her unwavering assistance and patience in answering my countless

emails. I also appreciate the support of Trystan Denhard, Jemma Natanson, and Ruth Yun.

                                                iv


--- PAGE BREAK ---

      Lastly, my deepest gratitude goes to my family, my mother and father, for their uncon-

ditional love, guidance, and support in all aspects of my life. Words cannot fully express my

appreciation. I also wish to thank my boyfriend, Bo He, who has embraced and lived with my

self-doubt while providing constant encouragement and companionship during my hardest times.

I include him in this section because, to me, he is already an irreplaceable part of my family.

      Finally, I acknowledge the many others who have supported me on this journey. While it is

impossible to name everyone, please know that your contributions have not gone unnoticed, and

I sincerely appreciate each and every one of you.




                                                v


--- PAGE BREAK ---

                                      Table of Contents



Dedication                                                                                        ii

Acknowledgements                                                                                  iii

Table of Contents                                                                                 vi

List of Tables                                                                                    ix

List of Figures                                                                                   xi

Chapter 1:       Introduction                                                                      1

Chapter 2:    Optimal Tour Length Estimation of the Traveling Salesman Problem                     6
   2.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .      6
   2.2 Extension in the TSPLIB . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .        10
   2.3 Experiments and Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .        13
         2.3.1 Euclidean TSP Instances . . . . . . . . . . . . . . . . . . . . . . . . . .        13
         2.3.2 Non-Euclidean TSP Instances . . . . . . . . . . . . . . . . . . . . . . .          17
         2.3.3 Performance of the stdRT Predictor on Different Instance Sets . . . . . .          24
   2.4 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .       25

Chapter 3:    Estimating Optimal Objective Values for the TSP, VRP, and Other Combi-
              natorial Problems using Randomization                                               27
   3.1   Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   27
   3.2   Using mean and std to Predict TSP Tour Length . . . . . . . . . . . . . . . . .          29
   3.3   Using mean and std to Predict VRP Tour Length . . . . . . . . . . . . . . . . .          34
   3.4   Using mean, std, and min to Predict VRP Tour Length . . . . . . . . . . . . . .          42
   3.5   Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     46

Chapter 4:  Optimal TSP Tour Length Estimation Using Sammon Maps                                  48
   4.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     48
   4.2 Estimation of Optimal Tour Length Using Sammon Maps . . . . . . . . . . . . .              50
   4.3 Results on São Paulo Road Maps . . . . . . . . . . . . . . . . . . . . . . . . . .        53
   4.4 Results on Random TSPs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .         57
   4.5 Results on Instances with Obstacles . . . . . . . . . . . . . . . . . . . . . . . .        59
   4.6 Results on 3DTSPs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .        60
   4.7 Further Thoughts and Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . .        65


                                                vi


--- PAGE BREAK ---

Chapter 5:  Estimating Optimal Split Delivery Vehicle Routing Problem Solution Values           68
   5.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   68
   5.2 Literature Review . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    71
   5.3 Modified Clarke & Wright Algorithm with Split Deliveries . . . . . . . . . . . .         74
   5.4 Regression Models and Experiment Results . . . . . . . . . . . . . . . . . . . .         76
   5.5 An Application . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     80
   5.6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     84

Chapter 6:    The Application of Machine Learning Models to Optimal TSP Tour Length
              Estimation                                                                        86
   6.1   Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 86
   6.2   Literature Review . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 88
   6.3   Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 89
         6.3.1 Feature Engineering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 90
         6.3.2 Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 94
         6.3.3 Computational Experiments . . . . . . . . . . . . . . . . . . . . . . . . 95
   6.4   Results and Discussions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 96
         6.4.1 Choosing m for Instance Size Reduction . . . . . . . . . . . . . . . . . . 96
         6.4.2 Model Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 98
         6.4.3 The Ability to Generalize . . . . . . . . . . . . . . . . . . . . . . . . . . 104
         6.4.4 Model Performance on the Entire Set . . . . . . . . . . . . . . . . . . . 108
   6.5   Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 110

Chapter 7:    Accurately Estimating Optimal SDVRP Solution Values using Regression
              Models                                                                            112
   7.1   Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 112
   7.2   Instance Generation and Benchmarking . . . . . . . . . . . . . . . . . . . . . . 115
   7.3   Regression Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 117
         7.3.1 Feature Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 117
         7.3.2 Model Development . . . . . . . . . . . . . . . . . . . . . . . . . . . . 119
   7.4   Results and Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 122
         7.4.1 Model Comparison Using Step-Wise Regression Features . . . . . . . . . 123
         7.4.2 Model Performance on Generated SDVRP Instances . . . . . . . . . . . 125
         7.4.3 Generalization to Size 70 and 90 Instances . . . . . . . . . . . . . . . . . 127
         7.4.4 Discussion and Future Directions . . . . . . . . . . . . . . . . . . . . . . 128
   7.5   Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 130

Chapter 8:    Conclusion and Future Directions                                                  132

Appendix A: Properties of Standard Deviation of Random Tour Lengths                             136

Appendix B: Mean and Standard Deviation for the MST and MWM                                 142
  B.1 Results for the MST . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 144
  B.2 Results for the MWM . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 146

Appendix C: Detailed SDVRP Instance Generation                                                  149

                                               vii


--- PAGE BREAK ---

References          151




             viii


--- PAGE BREAK ---

                                     List of Tables


2.1  Comparison between models on Euclidean TSPLIB Instances . . . . . . . . . . .            12
2.2  Euclidean TSPLIB Instances: log-log Scaled Regression . . . . . . . . . . . . .          12
2.3  Comparison between models on Randomly Generated Uniform Instances . . . .                14
2.4  Comparison between Models on Tn,m Instances . . . . . . . . . . . . . . . . . .          15
2.5  Tn,m Instances: log-log Scaled Regression . . . . . . . . . . . . . . . . . . . . .      16
2.6  Comparison among models on Euclidean Instances . . . . . . . . . . . . . . . .           17
2.7  Non-Euclidean Instances in TSPLIB . . . . . . . . . . . . . . . . . . . . . . . .        18
2.8  Randomly Generated Manhattan L1 Instances . . . . . . . . . . . . . . . . . . .          19
2.9  Comparison between models on Non-Euclidean Instances by Random Multipli-
     cation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   20
2.10 Comparison between models on Non-Euclidean Instances by Random Multipli-
     cation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   21
2.11 Comparison between models on Non-Euclidean Instances by Random Multipli-
     cation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   21
2.12 Comparison between models on Non-Euclidean Instances by Random Substitution              22
2.13 Comparison between models on Non-Euclidean Instances by Random Substitution              23
2.14 Comparison between models on Non-Euclidean Instances by Random Substitution              23
2.15 São Paulo Road Maps . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     24
2.16 Performance of the stdRT Predictor on Different Instance Sets . . . . . . . . . .        25

3.1 Non-Euclidean TSP Instances with Normally Distributed Distances . . . . . . . .           33
3.2 The meanRT and stdRT Model on TSP Instances . . . . . . . . . . . . . . . . .             34
3.3 The stdRT Model on TSP Instances . . . . . . . . . . . . . . . . . . . . . . . . .        34
3.4 Different Prediction Models in the Literature . . . . . . . . . . . . . . . . . . . .     35
3.5 Different Models on VRP Instances . . . . . . . . . . . . . . . . . . . . . . . .         37
3.6 Summary Statistics of meanCW i and stdCW i . . . . . . . . . . . . . . . . . . . .        37
3.7 Different Models on 10,000 VRP Instances . . . . . . . . . . . . . . . . . . . .          40
3.8 Different Models on 131 CVRPLIB Instances . . . . . . . . . . . . . . . . . . .           41
3.9 Different Models on all VRP Instances . . . . . . . . . . . . . . . . . . . . . . .       42
3.10 Different Models on the 10,000 VRP Instances . . . . . . . . . . . . . . . . . . .       45
3.11 Different Models on the 117 CVRPLIB Instances . . . . . . . . . . . . . . . . .          45
3.12 Different Models on the Joint Set with 10, 117 Instances . . . . . . . . . . . . . .     46

4.1   São Paulo Road Maps . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    54
4.2   Sampled São Paulo Road Maps . . . . . . . . . . . . . . . . . . . . . . . . . . .      55
4.3   Sampled
      √        4 km2 São Paulo Road Maps . . . . . . . . . . . . . . . . . . . . . . .       57
4.4     N ASammon on Random Instances . . . . . . . . . . . . . . . . . . . . . . . . .       58


                                            ix


--- PAGE BREAK ---

4.5 Properties of Random Instances . . . . . . . . . . . . . . . . . . . . . . . . . . .    59
4.6 Radius-1 disk removed . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     60
4.7 Radius-2 disk removed . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     61
4.8 Radius-3 disk removed . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     61
4.9 Radius-4 disk removed . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     62
4.10 Uniform 3DTSPs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     63
4.11 Gaussian 3DTSPs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    63
4.12 Uniform 3DTSPs with Central Cuboid Removed . . . . . . . . . . . . . . . . . .         64

5.1   Different Models on the Training and Test Sets . . . . . . . . . . . . . . . . . .    78
5.2   Different Models on the Full SDVRP Instance Set . . . . . . . . . . . . . . . . .     79
5.3   Application of Estimates as a Stopping Criteria . . . . . . . . . . . . . . . . . .   84

6.1   Result of Different Models on Instances of Size 20, 50, and 100 . . . . . . . . . . 99
6.2   Different Models Trained on Instances of Fixed Size . . . . . . . . . . . . . . . . 100
6.3   Out-of-Sample Evaluation Results . . . . . . . . . . . . . . . . . . . . . . . . . 108
6.4   Mixed Distribution MAPE Results (20% Training) . . . . . . . . . . . . . . . . 109
6.5   Mixed Distribution MAPE Results (80% Training) . . . . . . . . . . . . . . . . 109

7.1   MAPE results across regression models on generated SDVRP instances. . . . . . 125
7.2   MAPE results across regression models on size 70 and size 90 instances . . . . . 127

B.1   The stdST Model on MST Instances . . . . . . . . . . . . . . . . . . . . . . . . 145
B.2   The meanST and stdST Models on MST Instances . . . . . . . . . . . . . . . . 145
B.3   The stdM Model on MWM Instances . . . . . . . . . . . . . . . . . . . . . . . . 147
B.4   The meanM and stdM Models of MWM Instances . . . . . . . . . . . . . . . . 147




                                            x


--- PAGE BREAK ---

                                    List of Figures


3.1   The Multicollinearity between the meanCW and minCW Predictors . . . . . . .            46

4.1   A Road Map in São Paulo . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   51
4.2   Abstract of the Road Map . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   51
4.3   A Sammon Map of the Road Map . . . . . . . . . . . . . . . . . . . . . . . . .         52

5.1   Example Instance for VRP and SDVRP with demands D1, D2, and D3 . . . . . .             69
5.2   Merging two routes according to MCWSD . . . . . . . . . . . . . . . . . . . . .        76
5.3   EEMC&W on Two VRP Instances . . . . . . . . . . . . . . . . . . . . . . . . .          83
5.4   EEMC&W on P-n50-k8 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .       84

6.1   MAPE Values of Linear Regression Models . . . . . . . . . . . . . . . . . . . . 97
6.2   Comparing meanRT Predictors . . . . . . . . . . . . . . . . . . . . . . . . . . . 98
6.3   MAPE Values of Linear Regression Models . . . . . . . . . . . . . . . . . . . . 99
6.4   Visualization of the Pruned KAN Model Starting with Four Hidden Neurons . . . 102
6.5   Two Tn,m Instances . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 105
6.6   Two 2-Dimensional Triangular Distributed Instances and the Distribution . . . . 105

7.1   KAN Model Using Stepwise Regression Features . . . . . . . . . . . . . . . . . 123
7.2   Final KAN Model Using Stepwise Regression Features . . . . . . . . . . . . . . 124
7.3   KAN Model Using Selected Features by Feature Importance . . . . . . . . . . . 126

A.1 Distribution of Random Tour Lengths for 20 Vertices . . . . . . . . . . . . . . . 138
A.2 Variance in Random Tour Lengths versus Sum of Variance in Edge Lengths . . . 138

B.1 Example of MST and MWM . . . . . . . . . . . . . . . . . . . . . . . . . . . . 143




                                            xi


--- PAGE BREAK ---

Chapter 1:      Introduction



      The Traveling Salesman Problem (TSP) and the Vehicle Routing Problem (VRP) are fun-

damental combinatorial optimization problems with common applications in logistics and trans-

portation. Due to their NP-hard nature, solving these problems to optimality for large instances

is computationally infeasible. Consequently, researchers and practitioners often rely on heuris-

tic and approximate methods to obtain near-optimal solutions within a reasonable computational

time and an acceptable margin of error. In this thesis, we develop robust estimation models for the

TSP, VRP, and the Split Delivery Vehicle Routing Problem (SDVRP). Instead of solving these

problems and learn about the solution details, our models estimate the total tour length in the

optimal tour or the optimal set of routes.

      Having an accurate estimate of the optimal solution value is crucial for several reasons.

First, these estimates provide a benchmark to evaluate heuristic algorithms on new instances,

ensuring their solution quality. Additionally, in strategic planning, decision-makers often need

insights into routing costs without computing exact solutions or knowing instance details. For

example, when determining depot locations or service regions. Lastly, accurate estimators en-

hance the efficiency of hybrid optimization frameworks like Approximate Dynamic Programming

(ADP), where cost estimates guide the exploration of complex solution spaces more effectively.

      Several methods have been proposed to estimate the optimal tour length in routing prob-



                                                1


--- PAGE BREAK ---

lems. Traditionally, researchers have used topological predictors and statistical approximations,
                          √
such as the well-known        N A predictor for Euclidean TSP instances, with A representing the

area containing all vertices and N representing the total number of vertices (Beardwood, Halton,

& Hammersley, 1959). More recently, regression models and machine learning techniques have

demonstrated strong predictive performance in estimating tour lengths across different problem

variants, including TSP and VRP. This thesis explores statistical and machine learning approaches

to estimating optimal solution values for the TSP, VRP, and SDVRP, extending previous models

and proposing new methodologies.

      A notable advancement in tour length estimation is the incorporation of moment statistics

from random feasible tour lengths, developed by Basel and Willemain (2001). They revealed

a surprising yet powerful relationship between the optimal tour length and the standard devia-

tion of random feasible tour lengths. This approach has since been expanded in our early work

(Kou, Golden, & Poikonen, 2022), demonstrating that predictors derived from random feasible

solutions significantly enhance predictive accuracy across a broader class of problems, including

non-Euclidean TSP instances.

      Building on these insights, this thesis introduces novel methodologies that enhance predic-

tive accuracy through a combination of predictors obtained from heuristic-generated solutions

and topological predictors. For example, in order to obtain the moment predictors for the SD-

VRP, we develop a modified Clarke & Wright heuristic with split deliveries (MCWSD), which

efficiently explores the solution space.

      Furthermore, we systematically investigate the trade-offs between interpretability and ac-

curacy by comparing simple linear regression models to more complex machine learning tech-

niques, including random forests (RF), multilayer perceptrons (MLP), and Kolmogorov-Arnold

                                                 2


--- PAGE BREAK ---

Networks (KAN) (Liu et al., 2024). While RF and MLP models capture complex nonlinear rela-

tionships, they lack interpretability. In contrast, KAN offers a balance by providing interpretable

yet powerful models capable of learning nonlinear relationships while being able to extract an-

alytical expressions. We leverage KAN models to better understand the functional relationships

between features and the optimal solution values.

      The remainder of this dissertation is organized as follows. Chapter 2 extensively examines

linear regression models for the Traveling Salesman Problem (TSP). We update and extend com-

putational experiments by Basel and Willemain (2001). Next, we answer the question of why

does such a simple predictor (standard deviation in random tour lengths) work by revealing the
                                                                              √
relationship between the standard deviation predictor and the well-known          N A predictor for

Euclidean instances. We also empirically show that the standard deviation predictor is valid for

both Euclidean and non-Euclidean instances by applying it to the TSPLIB instances: randomly

generated instances and real-world instances.

      Chapter 3 shows that the strong power-law relationship between the standard deviation in

random feasible solution values and the optimal solution value also holds for other Euclidean and

near-Euclidean combinatorial optimization problems like the vehicle routing problem (VRP). We

then enhance the estimation ability of the model by considering a second predictor: the mean in

random feasible solution values. Experimental results show that by using the mean and standard

deviation, we can accurately predict the optimal solution values for the TSP and VRP. For the

VRP, the random feasible solutions are obtained from a modified Clarke and Wright heuristic

(Sinha Roy, Golden, Masone, & Wasil, 2023). In the second part of this chapter, we further

improve the linear regression model for estimating the optimal solution value for the VRP by

doing a small amount of extra work to include the minimum of the modified Clarke and Wright

                                                3


--- PAGE BREAK ---

heuristic solution values.

      Chapter 4 introduces the use of Sammon mapping (Sammon, 1969) as a feature extraction

method for non-Euclidean TSP instances, focusing particularly on real-world road map prob-

lems. We explore its effectiveness through computational experiments on road map data from

São Paulo, Brazil, as well as a simulated dataset, comparing the Sammon mapping-based model

with traditional circuity-factor-based approaches. Our analyses demonstrate that this approach

effectively estimates optimal tour lengths while providing the benefit of visualization.

      Chapter 5 addresses the Split Delivery Vehicle Routing Problem (SDVRP), extending our

methodologies developed for the VRP to this more complex problem. We propose a modified

Clarke & Wright algorithm that is adapted to handle split deliveries (MCWSD), outline the fea-

ture engineering process, and present a thorough evaluation of model accuracy and interpretabil-

ity. The experimental analyses consider established DIMACS benchmark instances, highlighting

the impressive accuracy with an error margin of approximately 3%. In this chapter, we also

provide an application of regression models on enhancing a heuristic solver for the VRP.

      Chapter 6 presents a machine learning-based approach for TSP tour length estimation,

leveraging linear regression (LR), random forests (RF), and neural networks (NNs), including

multilayer perceptron (MLP) and Kolmogorov-Arnold Networks (KAN). The KAN model is

trained on diverse TSP instances and exhibits strong generalizability compared to existing meth-

ods. An efficient feature engineering strategy is proposed to balance computational cost and pre-

dictive performance, making the models applicable to larger instances. Robustness is evaluated

on challenging datasets different from the training set, revealing that RF models achieve high ac-

curacy but struggle with extrapolation, while KAN models perform well on unseen distributions.

This work bridges the gap between traditional analytical methods and ML-based estimations,

                                                4


--- PAGE BREAK ---

providing a generalizable framework with promising accuracy.

      Chapter 7 focuses on the accurate estimation of the optimal total tour length in the SDVRP.

Using a newly generated dataset of 2,160 SDVRP instances, solved to near-optimality via the

SplitILS heuristic (Silva, Subramanian, & Ochi, 2015), we develop and evaluate multiple regres-

sion models. Specifically, we again compare the performance of linear regression, random forests

(RF), multilayer perceptrons (MLP), and Kolmogorov-Arnold Networks (KAN). Our findings in-

dicate that while machine learning models provide slight accuracy improvements, their benefits

over linear regression are marginal. A comparative analysis on the randomly generated instances

highlights the trade-off between model accuracy and interpretability. Ultimately, our results sug-

gest that linear regression remains a strong candidate for SDVRP optimal solution estimation due

to its interpretability and competitive performance, making it a practical choice for real-world

applications.

      In Chapter 8, we present the concluding remarks and briefly summarize our contributions.

We also identify potential directions for future research, emphasizing opportunities to extend

our methodologies to other combinatorial optimization problems and explore how high-accuracy

estimation models can be leveraged to develop heuristic algorithms for more complex routing

problems.




                                                5


--- PAGE BREAK ---

Chapter 2:      Optimal Tour Length Estimation of the Traveling Salesman Problem




2.1   Introduction

      The traveling salesman problem (TSP) is the most well-studied NP-hard problem in com-

binatorial optimization. Given a set V of vertices and known pairwise distances, the objective is

to find a tour with minimum total distance where all vertices in V are visited exactly once. Such

a tour is called an optimal tour, and the distance traveled in this tour is called the optimal tour

length. In this chapter, we estimate the optimal tour length without knowing the optimal tour.

      Although there are numerous optimal tour length estimation models in the literature, most

of them can only be applied to TSP instances under certain restrictions. To be specific, most

studies concentrate on estimating the optimal tour length in Euclidean TSPs. To our knowledge,

no studies have emphasized the estimation of optimal tour length in non-Euclidean TSPs, es-

pecially when the metric is unknown and we are only given a distance matrix. For example,

when the longitude and latitude data are not available in a real-world TSP or when we have a

machine scheduling problem, where the distance between (i, j) is the amount of reconfiguration

time between completing job i and starting job j.

      For example, Beardwood et al. (1959) formulated one of the earliest and most commonly

referenced models for identically, independently, and uniformly distributed vertices on a Lebesgue



                                                6


--- PAGE BREAK ---

set with a strictly positive measure in R2 . That is,


                                              L∗   p
                                      limN →∞ √ = β A0                                         (2.1)
                                               N


where L∗ denotes the optimal tour length, N is the number of vertices, β is a constant, and A0

is the measure of the Lebesgue set, which coincides with its area in two-dimensional Euclidean

space. Beardwood et al. (1959) derived the bounds on β for the Euclidean distance metric and

argued that 0.62 ≤ β ≤ 0.92 and β ≈ 0.75. Many researchers have subsequently studied the

value of β or derived tighter bounds on it for different types of instances. For vertices distributed

in rectangles, Daganzo (1984) studied the value of β for the Euclidean and Manhattan distance

metrics. The Manhattan distance between two vertices a = (x1 , y1 ) and b = (x2 , y2 ) is defined

by

                                    da,b = |x2 − x1 | + |y2 − y1 |.                            (2.2)


Daganzo (1984) derived upper bounds of β as 1.15 for the Manhattan distance metric and 0.90

for the Euclidean distance metric. Also considering the Euclidean distance metric, Chien (1992)

found the best fit β for instances with size 5 ≤ N ≤ 30 to be 0.82. The estimation of L∗ can also
                                                                       √
be improved by including other explanatory variables, in addition to       N A0 . For example, Chien

(1992) compared seven models and argued that the following model gives a robust estimation of

the value of L∗ :
                                                        p
                               L∗ = 2.1 ∗ R̄ + 0.67 ∗    (N − 1)A1                             (2.3)


where R̄ is the average distance between all other vertices and the starting vertex and A1 repre-

sents the area of the smallest rectangle that can cover all vertices. Subsequently, Kwon, Golden,


                                                  7


--- PAGE BREAK ---

and Wasil (1995) presented two neural network models that make accurate estimations of L∗ for

instances with 10 to 80 uniformly i.i.d. vertices, using the explanatory variables N, A1 , and the

ratio of the length to the height of the service region.

      Recently, models making accurate estimations of optimal tour length using instance size

N or properties of the distribution of vertices in two-dimensional Euclidean space have been

presented in the literature. For instance, Nicola, Vetschera, and Dragomir (2019) used forward

selection and backward elimination stepwise regression to build a model for optimal tour length

estimation that not only includes instance size N but also maximum distance across all pairs

of vertices, the sum of distances to the nearest neighbor for each vertex, variance in x and y

coordinates of vertices, and variance of distances to the average x-y coordinates. Using regres-

sion, Cavdar and Sokol (2015) discovered the following distribution-free model for estimation of

optimal tour length in the TSP:

                                                      s
                                                                             A2
                      q
            L∗ = 2.791 N (cstdevx ∗ cstdevy ) + 0.2669 N (stdevx ∗ stdevy )                     (2.4)
                                                                            c̄x c̄y


where cstdevx and cstdevy measure the standard deviations of the absolute distances to the ver-

tices from the horizontal and vertical midpoint lines of the rectangular space, stdevx and stdevy

are the standard deviations in x and y coordinates, c̄x and c̄y are the average distances of vertices

to the horizontal and vertical midpoint lines of the space, and A2 measures the area of the rect-

angular space. Although both Nicola et al. and Cavdar and Sokol obtained promising results in

instances where the vertices may not be uniformly distributed, the Euclidean distance between

pairs of vertices was still a key assumption. In fact, as long as the triangle inequality is satisfied

by the distance matrix, there exists a strong, predictive, and non-linear correlation between L∗


                                                  8


--- PAGE BREAK ---

              √
and mean −        N std where std and mean represent standard deviation and mean for all tour

lengths in the TSP (Sutcliffe, Solomon, & Edwards, 2012).

      In the paper that motivates our work, Basel and Willemain ran a regression on 17 Euclidean

instances from the TSPLIB repository (Reinelt, 1991) and introduced the optimal tour length

estimation model:

                               ln(L∗ ) = 1.798 + 0.927 ln(stdRT )                             (2.5)


where stdRT is the standard deviation of tour lengths for 20,000 random tours. Even though these

experiments focused on a small number of Euclidean instances from the TSPLIB, we note that

the computation of stdRT does not require the problem to be constructed in Euclidean space or

have any known metric. Thus, it may be possible to extend this estimation model to a wider range

of TSP instances.

      In most of the literature on this topic, as mentioned earlier, optimal tour length estimators

can only be computed for Euclidean instances, and it has been shown that strong correlations exist

between L∗ and N and between L∗ and the area over which vertices are distributed. Although

some researchers have studied instances with the Manhattan distance metric, we must know x-
                                                                                            √
y coordinates to estimate the optimal tour length. Since the value of the coefficient for       NA

varies for instances with different N values and different shapes, in this chapter, we choose to

fit the stdRT model on different sets of instances. We argue that accurate estimations using this

stdRT model are not restricted to Euclidean instances; in fact, the model also works well on

non-Euclidean instances.

      The rest of the chapter is organized as follows. In Section 2.2, we extend the experiment

presented by Basel and Willemain to more Euclidean instances in the TSPLIB, and we compare


                                                9


--- PAGE BREAK ---

                      √
the result with the       N A model for A defined as the area of the convex hull of all the vertices. In
                                                                                        √
Section 2.3, we present more experiments that compare the stdRT model and the            N A model in

terms of their predictive abilities on randomly generated Euclidean instances and hard-to-solve

tetrahedron instances introduced in Hougardy and Zhong (2018). In such tetrahedron instances,

the vertices are distributed on a two-dimensional projection of edges of a tetrahedron. To show, at

least partially, why the stdRT model makes sense, we demonstrate in the Appendix A that there
                                                                                          √
is an asymptotic linear relationship between the stdRT model and the widely used           N A model

in Euclidean space, under certain restrictions. We also present applications of the stdRT model

for non-Euclidean instances, including randomly generated instances and São Paulo road maps.

Lastly, in Section 2.4, we present some final remarks.



2.2    Extension in the TSPLIB

      Since the estimation model on 17 instances constructed by Basel and Willemain (2001)

has an r-squared value of 0.99 and a mean absolute percentage error (MAPE) of 15.5%, we first

want to verify whether a similar model is valid on a wider selection of Euclidean instances in

the TSPLIB. According to the authors, using 100 random tours instead of 20,000 to compute

stdRT only increases the MAPE of the stdRT model by 0.9%, and the value of stdRT converges

as the number of random tours increases (Basel & Willemain, 2001). In Section 2.3.1.1, we also

conducted a small experiment to show that considering 1000 random tours will lead to slightly

better estimation results than 100 random tours. Based on this result and our computational con-

straints, we choose to generate 1000 random tours for 76 Euclidean instances in the TSPLIB with
                                                                             √
size N ≤ 14, 051. In addition, we are also interested to see how the          N A model performs on



                                                     10


--- PAGE BREAK ---

Euclidean TSPLIB instances. Because we adopt the definition of A as the area of a convex hull,

we used the ConvexHull function from Scipy (Virtanen et al., 2020) to compute the correspond-

ing areas. For the optimal tour length for each instance that we want to estimate, we use the

best known solutions for these instances in the TSPLIB. In order to compare two predictors, we

compute the r-squared value and the MAPE. The r-squared value measures the goodness of fit of

the linear regression model, and the MAPE describes how good the predicted results are. Using

information gathered from the instances, we use linear regression to fit the following models and

find the numeric values for the constants a,b,c, and d:


                                            √
                                      L∗ = a N A + b                                         (2.6)

                                  ln(L∗ ) = c ln(stdRT ) + d.                                (2.7)



Note that if we exponentiate both sides of the second equation above with base e, we obtain



                                        L∗ = ed × stdRT c .                                  (2.8)



Thus, a strong linear relationship between ln(stdRT ) and ln(L∗ ) implies a strong power-law rela-

tionship between stdRT and L∗ . Table 2.1 summarizes the two above models and their estimation

results on the 76 Euclidean instances in the TSPLIB mentioned above. Note that for the log-log

model, we compute its MAPE by exponentiating the predicted ln(L∗ ) and then comparing it with

the true L∗ . Thus, we can compare the MAPE of both models in Table 2.1 directly.

      The results show that, when extending Basel and Willemain’s stdRT model to more in-

stances in the TSPLIB, the estimation result is still valid with an r-squared value of 0.97 and an


                                                11


--- PAGE BREAK ---

                                                                             √
                ln(L∗ ) = 1.656 + 0.927 ln(stdRT )      L∗ = 0.541 + 9992.58 N A
                         R-squared: 0.97                     R-squared: 0.99
                         MAPE: 27.88%                        MAPE: 149.25%




             Table 2.1: Comparison between models on Euclidean TSPLIB Instances



MAPE of 27.88%. The MAPE of the stdRT model on 76 instances is greater than the MAPE

that Basel and Willemain (2001) reported on 17 instances. However, the r-squared value is still

very high, and this validates the strong power-law relationship between the standard deviation

of random tour length and the optimal tour length. The small MAPE indicates that the value
                                                                                  √
predicted by the stdRT estimator is more accurate than the one predicted by the    N A estimator.
                                      √
The r-squared value of 0.99 for the       N A model confirms the strong linear relationship between
√
    N A and the optimal tour length. Since, in our experiment set, instances range from 52 to
                                                   √
14,051 vertices, we find that the MAPE for the      N A model can be lowered by taking a log-log

transformation. The result is shown in Table 2.2. Despite the fact that instances vary largely in
                                                          √
                               ln(L∗ ) = 0.320 + 0.933 ln( N A)
                                        R-squared: 0.98
                                        MAPE: 19.48%




               Table 2.2: Euclidean TSPLIB Instances: log-log Scaled Regression




                                                   12


--- PAGE BREAK ---

size and in the distribution of vertices, for Euclidean instances in the TSPLIB, both the stdRT
          √
and the       N A models obtain high r-squared values and low MAPE values, especially when we

consider the log-log scaled regression.



2.3     Experiments and Results


2.3.1     Euclidean TSP Instances
                                                                  √
        In the previous section, we observe that both stdRT and       N A can predict L∗ with rela-

tively high accuracy. In the Appendix A, we show that, when vertices are uniformly distributed

in a finite space, the random tour lengths converge to a normal distribution. In addition, if ver-
                                                                      √
tices are uniformly distributed in a unit square or disk, stdRT and    N A are linearly dependent
                                                                                  √
as N converges to infinity. Based on the connection between the stdRT and          N A estimators

when vertices are uniformly i.i.d., we ran a number of experiments on different sets of Euclidean

TSP instances to compare the estimation results using the two different explanatory variables.

For these Euclidean instances, experiments similar to those described in Section 2.2 were con-

ducted. For instances with a known optimal tour length, the values were used directly. Otherwise,

we fit two estimators to approximate the near optimal tour length, applying the Lin-Kernighan-

Helsgaun heuristic (LKH). The LKH heuristic is known to produce a tour length within about 1%

of optimal, and as a heuristic, it is less time consuming than finding exact solutions (Applegate,

Bixby, Chvátal, & Cook, 2011). As discussed in the introduction, we fit the linear models for

different sets of instances independently for model comparison, since size and distribution of

instances vary largely among different sets.




                                               13


--- PAGE BREAK ---

2.3.1.1    Randomly Generated Uniform Instances

      On 1 × 1, 2 × 2, and 5 × 5 squares we randomly generated 10 instances with 10 uniformly

distributed vertices, 10 instances with 20 uniformly distributed vertices, and 10 instances with

50 uniformly distributed vertices. For these 90 instances with uniformly distributed vertices, we
                                                                                        √
provide a comparison between the two models in Table 2.3. We found that stdRT and           N A both
                                                                       √
              ln(L∗ ) = 1.506 + 0.940 ln(stdRT ) L∗ = 1.564 + 0.904 N A
                       R-squared: 0.95                  R-squared: 0.96
                        MAPE: 12.33%                     MAPE: 15.43%




      Table 2.3: Comparison between models on Randomly Generated Uniform Instances


                                                                    √
provide good results for fitting instances in this set, where the    N A estimator reaches a higher

r-squared value but the MAPE of the stdRT estimator is lower. In addition to the stdRT estimator

with 1000 random tours, we also experimented with the stdRT estimator with 100, 500, 5000,

and 10000 random tours. The r-squared values of models increase from 0.93 for 100 random

tours to 0.95 for 1000 and more random tours. However, the MAPEs decrease from 14.80% for

100 random tours to 12.33% for 1000 random tours and then fluctuate around 12.33%. Based on

this observation, we decided to generate 1000 random tours for all subsequent experiments.




                                                14


--- PAGE BREAK ---

2.3.1.2      Tn,m Instances

        Hougardy and Zhong (2021) introduced hard-to-solve Euclidean TSP instances called Tn,m .

These instances with different values of n, m are called tetrahedron instances, where n and m

correspond to the number of vertices along the side and inside the projection of a tetrahedron

(Hougardy & Zhong, 2021). In their paper, the authors show that for Tn,m instances, it is time-

consuming to find the exact optimal solution using Concorde (Hougardy & Zhong, 2021). We

used 50 instances posted on the authors’ website with the optimal tour costs, and we found the

estimation results to be extremely accurate for this set of instances using both models. These

instances satisfy 50 < N < 200,


                                           
                                3N − 40                       N +2
                           n :=                 and    m :=        − n.                        (2.9)
                                  10                            3


A summary of this comparison can be found in Table 2.4. For both models, we obtained an r-

squared value of 0.99 and low MAPEs. In fact, if we consider the log-log scaled regression using
                                                                            √
             ln(L∗ ) = 4.566 + 0.730 ln(stdRT ) L∗ = 401217.773 + 0.557 N A
                       R-squared: 0.99                   R-squared: 0.99
                       MAPE: 1.34%                        MAPE: 3.85%




                     Table 2.4: Comparison between Models on Tn,m Instances


√
    N A as the explanatory variable, data points in the set are fit almost perfectly and the MAPE is


                                                 15


--- PAGE BREAK ---

significantly reduced, which is shown in Table 2.5. Both estimation models fit surprisingly well
                                                          √
                               ln(L∗ ) = 3.284 + 0.757 ln( N A)
                                        R-squared: 0.99
                                         MAPE: 0.59%




                       Table 2.5: Tn,m Instances: log-log Scaled Regression



for Tn,m instances. This may be explained by the consistency of structures for instances in the

set. In addition, based on the scatterplot and the fitted line, we posit that if we generate larger

Tn,m with different values of m and n, the corresponding data points will continue to fall on or

very close to the fitted regression lines.



2.3.1.3       Comparison with a Method from the Literature

      Recall equation (2.4) by Cavdar and Sokol (2015) presented in Section 2.1. To demonstrate

the performance of our stdRT model, we compared the r-squared value and MAPE of equation
             √
(2.4), the    N A, and the stdRT model in Table 2.6. For Cavdar and Sokol’s model from equa-

tion (2.4), instead of using their provided constant coefficients, we fit predictors using linear

regression and obtained new coefficients for different sets of instances. In Table 2.6, the columns

represent different models. Note that we run log-log regression for the stdRT model, while no
                                                         √
transformation is applied for the Cavdar and Sokol and       N A models. The first row contains the

results for Euclidean TSPLIB instances discussed in Section 2.2, while the second row covers the


                                                16


--- PAGE BREAK ---

results for randomly generated instances with uniformly distributed vertices from Section 2.3.1.1,

and the third row is for Tn,m instances from Section 2.3.1.2. It is not surprising that the model

introduced by Cavdar and Sokol (2015) does well on the three sets, except for its MAPE in row

1. However, it is worth noting that in order to use the model given by equation (2.4), one needs
                                                                         √
to compute 8 different variables. On the other hand, the traditional         N A model and the stdRT

model require that only two and one, respectively, be computed. In fact, the predictive results

given by the simple models are not far from the one offered by the more complicated model.
                                                               √
             Instance Set           Cavdar and Sokol             NA             stdRT
                                      r-squared: 0.99     r-squared: 0.99  r-squared: 0.97
         Euclidean TSPLIB
                                     MAPE: 174.72% MAPE: 149.25% MAPE: 27.88%
                                      r-squared: 0.98     r-squared: 0.96  r-squared: 0.95
   Randomly Generated Uniform
                                      MAPE: 8.25%         MAPE: 15.43% MAPE: 12.33%
                                      r-squared: 0.99     r-squared: 0.99  r-squared: 0.99
                 Tn,m
                                      MAPE: 2.23%         MAPE: 3.85%      MAPE: 1.34%

                  Table 2.6: Comparison among models on Euclidean Instances


                                                     √
        We also try to combine the stdRT and          N A estimators on Euclidean instances to see

whether a combination of two estimators can offer a higher predictive accuracy. However, we
                                               √
found that for our test instances, stdRT and       N A are linearly dependent. Also, when we compare

the relative errors of both models, the relative errors are never negatively correlated. Thus, in-

cluding both predictors will introduce multicollinearity, which complicates the estimation model

without offering a substantial increase in predictive accuracy.



2.3.2     Non-Euclidean TSP Instances

        Of course, we do not always operate in Euclidean space. For example, if we consider a

more realistic model of the TSP with multiple geographical locations as vertices and the roads

                                                     17


--- PAGE BREAK ---

between locations as edges, then the pairwise distance between vertices is, typically, not the

length of the straight line connecting the two vertices. Additionally, it is common to see a TSP

with only a distance matrix provided and no x-y coordinates of vertices. For non-Euclidean TSP
                                                                    √
instances, if we do not know the x-y coordinates of vertices, the    N A model is no longer valid,

since we are unable to compute A. For these sets of instances, we conducted our experiments by

testing the estimation ability of the explanatory variable stdRT . The following results show that

stdRT can still be used to produce relatively accurate estimations for some sets.



2.3.2.1    TSPLIB

      For 25 non-Euclidean instances from the TSPLIB with 17 to 666 vertices, we repeated our

process of finding the value of stdRT using 1000 random tours, and we tried to fit the estimation

model using the known optimal tour length. Table 2.7 displays the results. Although the MAPE

                               ln(L∗ ) = 2.655 + 0.843 ln(stdRT )
                                        R-squared: 0.86
                                        MAPE: 49.60%




                         Table 2.7: Non-Euclidean Instances in TSPLIB



of the stdRT estimator is relatively high for this dataset, we recall that the TSPLIB is a set in

which instances vary largely in size and in distribution of vertices. We can see a handful of data

points fall far away from the regression line while the others present a nice linear trend. From


                                               18


--- PAGE BREAK ---

the r-squared value of 0.86, we can conclude that there is still a strong power-law relationship

between stdRT and the optimal tour length.



2.3.2.2    Randomly Generated Instances with Manhattan L1 Distance

      For the 90 instances mentioned in Section 2.3.1.1 and using the same x-y coordinates,

we created 90 new instances with the Manhattan distance metric. Suppose we want to predict

the optimal tour length using only the distance matrix but not vertex coordinates. For these 90

random instances, we fit a stdRT estimator to predict the near optimal tour length obtained using

the LKH heuristic. Table 2.8 shows the outcome of this experiment. For these non-Euclidean

                               ln(L∗ ) = 1.440 + 0.933 ln(stdRT )
                                        R-squared: 0.95
                                        MAPE: 12.38%




                   Table 2.8: Randomly Generated Manhattan L1 Instances



instances with Manhattan L1 distances, stdRT appears to be a good predictor, with an r-squared

value of 0.95 and an MAPE of 12.38%.



2.3.2.3    Non-Euclidean Instances by Random Multiplication

      In order to have more instances on which to test the estimation potential of stdRT , we

randomly generated instances with 11, 12, 13, ..., 100 vertices, which had uniformly distributed


                                               19


--- PAGE BREAK ---

coordinates on a 1 × 1 square. Using the coordinates of vertices, we defined the following non-

Euclidean distance by random multiplication:



                                 drm (vi , vj ) = dEU C (vi , vj ) ∗ ai,j                    (2.10)



where dEU C (vi , vj ) is the Euclidean distance between two vertices vi and vj , and where ai,j are

independently and identically distributed random variables from a uniform distribution U (1 −

ϵ, 1 + ϵ) for some fixed ϵ. We generated such sets of 90 instances for different values of ϵ. It is

clear that the greater the value of ϵ, the less Euclidean the instances are. In Tables 2.9 to 2.11,

we include the fittings of the stdRT model for ϵ = 0.1, 0.2, 0.5, 0.7, 0.8, and 0.9. As we can

observe from these tables, as the instance sets become less Euclidean, the predictive ability of

stdRT diminishes. For instances close to Euclidean, the stdRT predictor has an r-squared value

of around 0.9 and an MAPE of around 6%.

            ln(L∗ ) = 1.459 + 0.726 ln(stdRT )          ln(L∗ ) = 1.400 + 0.754 ln(stdRT )
                     R-squared: 0.90                             R-squared: 0.89
                      MAPE: 5.77%                                 MAPE: 6.66%




Table 2.9: Comparison between models on Non-Euclidean Instances by Random Multiplication




                                                   20


--- PAGE BREAK ---

            ln(L∗ ) = 1.171 + 0.718 ln(stdRT )        ln(L∗ ) = 0.822 + 0.783 ln(stdRT )
                     R-squared: 0.88                           R-squared: 0.88
                      MAPE: 6.50%                               MAPE: 6.76%




Table 2.10: Comparison between models on Non-Euclidean Instances by Random Multiplication


            ln(L∗ ) = 0.714 + 0.665 ln(stdRT )        ln(L∗ ) = 0.539 + 0.548 ln(stdRT )
                     R-squared: 0.79                           R-squared: 0.80
                      MAPE: 9.06%                               MAPE: 7.15%




Table 2.11: Comparison between models on Non-Euclidean Instances by Random Multiplication




2.3.2.4    Non-Euclidean Instances by Random Substitution

     In a process similar to that in the previous section, we randomly generated instances with

11, 12, 13, ..., 100 vertices, which ranged from close to Euclidean to far from Euclidean. As

before, we started by randomly generating vertices with uniformly distributed coordinates on a

1 × 1 square. Using the coordinates of vertices, we defined the following non-Euclidean distance




                                                 21


--- PAGE BREAK ---

by random substitution:

                                         
                                         
                                         dEU C (vi , vj )    with probability 1 − θ
                                         
                                         
                      drs (vi , vj ) =                                                            (2.11)
                                         
                                         
                                         dEU C (vm , vn )
                                         
                                                               with probability θ


      where dEU C (vi , vj ) is the Euclidean distance between two vertices vi and vj , and (vm , vn )

is a random pair of vertices from V that is not the same pair as vi and vj . In other words, we

randomly replaced the edge length connecting two vertices with a random edge from the graph

with probability θ. For a fixed value of θ, 90 instances were randomly generated. If we used a

smaller θ, the instances in the set are closer to Euclidean instances. In Tables 2.12 to 2.14, we

present the performance of the stdRT model for θ = 0.1, 0.2, 0.5, 0.7, 0.8, and 0.9. Similar to the

case of random multiplication, we see that as we deviate from Euclidean instances, the predictive

ability of stdRT diminishes. Specifically, as θ increases from 0.1 to 0.9, r-squared values decrease

from 0.9 to around 0.8. Still, this is not a bad fit. Note that the prediction result of the θ = 0.9 set

is slightly better than the one for θ = 0.8; it might be the case that, after we reach a threshold of

high θ, instances have a mostly random distance matrix and lose their Euclidean structure.

             ln(L∗ ) = 1.432 + 0.725 ln(stdRT )              ln(L∗ ) = 1.352 + 0.791 ln(stdRT )
                      R-squared: 0.91                                 R-squared: 0.90
                       MAPE: 5.93%                                     MAPE: 6.58%




Table 2.12: Comparison between models on Non-Euclidean Instances by Random Substitution




                                                        22


--- PAGE BREAK ---

            ln(L∗ ) = 1.308 + 0.812 ln(stdRT )        ln(L∗ ) = 1.246 + 0.879 ln(stdRT )
                     R-squared: 0.88                           R-squared: 0.84
                      MAPE: 8.14%                               MAPE: 9.62%




Table 2.13: Comparison between models on Non-Euclidean Instances by Random Substitution


            ln(L∗ ) = 1.349 + 0.706 ln(stdRT )        ln(L∗ ) = 1.317 + 0.774 ln(stdRT )
                     R-squared: 0.78                           R-squared: 0.80
                      MAPE: 9.77%                               MAPE: 9.78%




Table 2.14: Comparison between models on Non-Euclidean Instances by Random Substitution




2.3.2.5    São Paulo Road Maps
                                                           √
     Merchán and Winkenbach (2019) validated the              N A model for road map instances in

São Paulo, and for one square kilometer segments, their model using linear regression generates

MAPEs of between 7.5% and 8.8% and an r-squared value of 0.94. These instances are non-

Euclidean, but they are nearly Euclidean. We considered one square kilometer road map seg-

ments in São Paulo with centers having latitude ∈ {−23.50, −23.51, ..., −23.69} and longitude

∈ {−46.75, −46.74, ..., −46.46}. We processed these 600 instances using the OSMnx module

(Boeing, 2017). Among these asymmetric TSP instances, we randomly selected 10 instances


                                                 23


--- PAGE BREAK ---

with 20 ≤ N < 40, 10 instances with 40 ≤ N < 60, ..., and 10 instances with 200 ≤ N < 220.

For each of these 100 selected instances, we generated 1000 random tours and computed their

near-optimal tour lengths using the LKH heuristic. Table 2.15 shows the result of fitting the

stdRT model to São Paulo road map instances. Based on the relatively high r-squared value, we

observe a strong power-law relationship between L∗ and stdRT . One reason that the estimation
                                             √
accuracy is not as high as the one for the       N A model in Merchán and Winkenbach (2019) may

be that we did not separate our instances into different sets based on their topological features, as

did Merchán and Winkenbach.

                                ln(L∗ ) = 1.416 + 0.966 ln(stdRT )
                                         R-squared: 0.86
                                         MAPE: 17.98%




                                Table 2.15: São Paulo Road Maps




2.3.3     Performance of the stdRT Predictor on Different Instance Sets

        To summarize our experiments, we provide the results on each instance set in Table 2.16.

We note that the estimation results of using the stdRT predictor are generally better for Euclidean

instances (the first three rows), where the r-squared value for fitted models for Euclidean instances

is at least 0.95. For non-Euclidean instances, although the predictive results are not as good as

for Euclidean instances, we still observe a moderate to strong power-law relationship between


                                                   24


--- PAGE BREAK ---

                           Instance Sets               R-squared       MAPE
                        Uniform Instances                 0.95         12.33%
                          Tn,m Instances                  0.99          1.34%
                  Euclidean TSPLIB Instances              0.97         27.88%
                       L1 Metric Instances                0.95         12.38%
                Non-Euclidean TSPLIB Instances            0.86         49.60%
                Random Multiplication Instances        0.79-0.90    5.77%-9.06%
                 Random Substitution Instances         0.78-0.91    5.93%-9.78%
                 São Paulo Road Maps Instances           0.86         17.98%

           Table 2.16: Performance of the stdRT Predictor on Different Instance Sets



the stdRT predictor and L∗ . In fact, we also observe that the stdRT predictor works better at

predicting L∗ when instances are closer to Euclidean instances.



2.4   Conclusion

      Although previous studies have developed multiple predictors for the optimal tour length,

most of them have focused on Euclidean instances or Manhattan instances. In this chapter, we

considered both Euclidean and non-Euclidean instances. We began by extending the work of

Basel and Willemain (2001) in the sense that we used 76 Euclidean instances from the TSPLIB,

ranging in size up to N = 14, 051, in our computational experiments. We found that compared
                    √
to the well-known       N A predictor, the predictor stdRT works nearly as well. We then show that
                                                                    √
as N → ∞, for Euclidean instances with uniformly i.i.d. vertices,       N A and stdRT have a linear

relationship. This helps explain why the stdRT predictor looked so promising in the paper by

Basel and Willemain (2001). Furthermore, we fit the stdRT predictor to non-Euclidean instances

and observe that for Euclidean instances with small perturbations stdRT predicts the optimal

tour length fairly well. Although this predictive ability diminishes as we move to instances that

violate the Euclidean property, it is still a reasonable predictor considering the fact that we do


                                                 25


--- PAGE BREAK ---

not have predictors of the optimal tour length for most non-Euclidean instances. Lastly, the real-

world application of the stdRT predictor on São Paulo road maps confirms its predictive ability

on real-world instances.




                                               26


--- PAGE BREAK ---

Chapter 3:      Estimating Optimal Objective Values for the TSP, VRP, and Other

                Combinatorial Problems using Randomization




3.1   Introduction

      As we discussed in the previous chapter, most works on optimal traveling salesman problem

(TSP) tour length estimation considered instances where vertices have known x-y coordinates and

the distance between vertices is Euclidean.

      These include more recent works of Nicola et al. (2019) using stepwise regression and built

a model including instance size N , maximum distance across all pairs of vertices, and the sum of

distances to the nearest neighbor for each vertex. Akkerman, Mes, and Heijnen (2020) considered

21 different structural properties from the literature and 11 newly proposed structural properties

before the feature selection step. Then, they used linear regression, random forest regression,

and neural networks to construct multiple models. The best models developed by Cavdar and

Sokol (2015), Nicola et al. (2019), and Akkerman et al. (2020) obtained promising results, with

r-squared values higher than 0.95 on their test instances.

      Recall that Basel and Willemain (2001) ran a linear regression on 17 Euclidean instances

from the TSPLIB repository (Reinelt, 1991) and introduced the following optimal tour length




                                                27


--- PAGE BREAK ---

estimation model:

                                ln(L) = 1.798 + 0.927 ln(stdRT )                              (3.1)


where stdRT is the standard deviation of tour lengths for 20,000 random tours. If we exponentiate

both sides of Equation (3.1), we obtain L = e1.798 stdRT 0.927 . Thus, this model describes the

power-law relationship between stdRT and L. In addition, noting that the computation of stdRT

does not require the problem to be formulated in Euclidean space or have any known metric, we

previously extended this estimation model to a wider range of TSP instances in Kou et al. (2022).

In related work, Kou, Golden, and Poikonen (2023) examined optimal tour length estimation

for non-Euclidean TSPs using Sammon mapping. While that model performs well on near-

Euclidean instances such as roadmap instances, its estimation accuracy diminishes as instances

become more structurally non-Euclidean.

      The estimates provided by the stdRT predictor on L make us wonder whether the power-law

relationship described in Equation (3.1) also holds for other combinatorial optimization problems

and how the predictive results can be improved. In Appendix B, we, therefore, show that for two

combinatorial optimization problems, which are solvable in polynomial time, the minimum span-

ning tree (MST) and maximum weight matching (MWM) problems, there is a strong power-law

relationship between the standard deviation in feasible solutions and the optimal solution if the

instances are Euclidean or near-Euclidean. These experimental results motivate us to generalize

the idea of using random feasible solution values to estimate the optimal objective values to other

NP-hard problems that lack known polynomial solvers.

      In our models presented in this section, we draw on Sutcliffe et al. (2012) who ran exper-

iments showing that for the TSP, when the distance matrix satisfies the triangle inequality, there


                                                28


--- PAGE BREAK ---

                                                     √
exists a strong correlation between L and mean −         N std, where std and mean are the standard

deviation and mean for all tour lengths in the TSP. Thus, we also consider including the predic-

tor meanRT , which is the average length of random tours, to potentially improve our estimation

model.

      We structure the rest of this chapter as follows. In Section 3.2, we enhance the estimation

model by including meanRT as another predictor and present experimental results for the TSP.

In Section 3.3, we discuss the results of the original model using only the standard deviation

predictor and this enhanced model to predict optimal capacitated vehicle routing problem (VRP)

costs by considering random heuristic solutions. We also recognize that, by adding the minimum

of 1,000 modified Clarke and Wright solution values as a third predictor, we can improve the

prediction. In Section 3.4, we provide a summary of models in the literature, describe our ex-

periments, and present the results. We also compare them to the results from our previous model

with only the mean and standard deviation predictors. Conclusions are then presented in Section

3.5. In the Appendix B, as mentioned earlier, we provide the experimental results for the MST

and MWM.



3.2    Using mean and std to Predict TSP Tour Length

      In the previous chapter, we showed that using the stdRT predictor, one can obtain similar
                                                             √
estimation results to those obtained using the traditional    N A predictor on Euclidean instances.

For two out of three sets of Euclidean instances that we tested, the model using only the stdRT

predictor even outperforms the model proposed by Cavdar and Sokol in Equation (2.4), which

considers eight different instance features (Kou et al., 2022). We also tested the performance



                                                29


--- PAGE BREAK ---

of the stdRT predictor on non-Euclidean instances and obtained reasonable estimates. A natural

next step is to test whether adding the mean of random tour lengths as another predictor can

enhance the predictive ability of the model. In this section, we use stdRT and meanRT collected

from 1,000 random tours and L solved using the LKH heuristic to study the results of fitting

Equation (3.2) to different TSP instances:



                               L = a1 × meanRT + b1 × stdRT + c1 .                            (3.2)




      It is worth noting that we do not apply log transformations in this model. In fact, we did

experiment with a variant of Equation (3.2) with a log transformation applied to each of the

three variables, and we found that the transformation improved the predictive results when the

dataset had a wider range in N . However, such a log transformation is not necessary for the

experiments included in this section. Table 3.2 provides the fitted coefficients and the predictive

results of Equation (3.2) on some test instance sets. Uniform, Arcsine, and Normal are three sets

of Euclidean TSP instances with sizes N = 10, 11, ..., 99, and the name of each describes its

distribution of vertices on the unit square. The multiplication set includes uniformly distributed

instances with size N = 30, 40, ..., 100 and distance defined by random multiplication for ϵ =

0.1, 0.2, ..., 0.9 as follows. For each pair of distinct vertices vi and vj ∈ V :



                                  drm (vi , vj ) = dEU C (vi , vj ) ∗ ai,j                    (3.3)



where dEU C (vi , vj ) is the Euclidean distance between two vertices vi and vj and where ai,j are



                                                    30


--- PAGE BREAK ---

i.i.d. random variables from a uniform distribution U (1−ϵ, 1+ϵ) for fixed ϵ. Similarly, we define

the non-Euclidean distance by random substitution between two vertices vi and vj :

                                        
                                        
                                        dEU C (vi , vj )   with probability 1 − θ
                                        
                                        
                     drs (vi , vj ) =                                                         (3.4)
                                        
                                        
                                        dEU C (vm , vn )
                                        
                                                             with probability θ.


That is to say, we randomly replace the edge length connecting two vertices with a random (other)

edge length from the graph with probability θ. The substitution set includes uniformly distributed

instances of the same size, N = 30, 40, ..., 100, and with distance defined by random substitu-

tion for θ = 0.1, 0.2, ..., 0.9. In the same way we generate random multiplication instances, we

replace all edge lengths that are above 0.8 with 0.8, and we call these instances the truncated

multiplication set. The second to last row of Table 3.2 contains all instances included in the

random multiplication and random substitution sets. In the last row, instances with normally

distributed distances are instances that do not have vertices located in an x-y coordinate system;

rather, we generate non-diagonal entries in the distance matrix according to a normal distribution

N (100, α). We generate 50 instances in this set with size N = 50 and α ranging from 0, 1, ..., 49.

Thus, the instances with normally distributed distances do not have any metric structure.

      Using these normally distributed distance matrices, the objective is to find the optimal

TSP tour length. In fact, in addition to stdRT and meanRT predictors, we also tried including the

minimum random tour length as another predictor. However, we found that the minimum random

tour length predictor is almost always not significant and is correlated with a linear combination

of stdRT and meanRT , so we excluded this predictor from further consideration. For example,

the multiplication set in Table 3.2 has the lowest adjusted r-squared value. When we add the


                                                       31


--- PAGE BREAK ---

minimum of 1,000 random tour lengths as another predictor, the adjusted r-squared value only

increases by 0.001 to 0.963 and the MAPE decreases to 5.194%. We also note that the p-value for

the minimum random tour length predictor is 0.209. Hence, we do not include it as a predictor in

our model.

      For comparison, we also include the results of model (3.1), which comes from our previous

paper (Kou et al., 2022). In Table 3.3, we present the fitted coefficients for ln(stdRT ) and the

constants as well as the predictive results of model (3.2) on the test instance sets from Table 3.2:



                                    ln(L) = b2 ln(stdRT ) + c2 .                                (3.5)



Combining the results from Table 3.2 and Table 3.3, we observe that using model (3.2), we can

predict the value of L reasonably well when TSP instances are Euclidean. On the other hand, for

non-Euclidean instances, using only stdRT as a predictor seems to be insufficient, and adding the

meanRT predictor can always increase the r-squared value and decrease the MAPE. For instance

sets that use stdRT alone and yield relatively poor results, like the multiplication set, the addition

of the meanRT predictor boosts the predictive results.

      We also note that L is positively correlated with meanRT and negatively correlated with

stdRT , which is intuitive since a higher standard deviation in random tours means that there is

a higher probability of finding shorter tours. In particular, we note that in the last row of Table

3.3, the coefficient for the ln(stdRT ) predictor is negative. In fact, if we do not take a log-

log transformation in this case, the obtained regression model has an adjusted r-squared value

of 0.991 and an MAPE of 5.817% as shown in Table 3.1. Recall that, for instances in this

set, the non-diagonal entries of the distance matrices are generated using normal distributions


                                                 32


--- PAGE BREAK ---

N (100, α) for different values of α. With the expected random tour length being the same for all

instances, larger α leads to a higher probability of finding shorter edges to complete the optimal

tour. Thus, we can observe that L is negatively correlated with stdRT . In this case, adding the

meanRT predictor (see last row of Table 3.2) improves the predictive result only slightly. On

the other hand, for the remaining instance sets, using only the meanRT instead of the stdRT

predictor yields better predictions. Thus, to form a model that can generalize to different types

of instances, it is necessary to include both predictors. For the substitution set, the only case in

which the coefficient for stdRT is positive in Table 3.2, the stdRT predictor in the model is not

significant.

                                 L = −13.626stdRT + 4957.529
                                      R-squared: 0.991
                                       MAPE: 5.817%




         Table 3.1: Non-Euclidean TSP Instances with Normally Distributed Distances



      In Appendix B, we demonstrate that random feasible solutions to the MST and MWM prob-

lems can, similarly, be used to estimate the optimal solution values for these easier combinatorial

optimization problems.




                                                33


--- PAGE BREAK ---

                            Variable Coefficients
                                                    meanRT      stdRT         constant       Adj.R2    MAPE
 Instance Set
                                                         ∗∗             ∗∗∗          ∗∗∗
                      Uniform                        0.718     −4.529      2.969             0.985     5.924%
                       Arcsine                       0.629∗∗∗ −5.898∗∗∗    5.383∗∗∗          0.977     6.612%
                                                          ∗∗∗        ∗∗∗
                       Normal                        0.995     −0.279      0.173             0.989     4.298%
                   Multiplication                    0.639∗∗∗ −3.019∗∗∗    3.570∗∗∗          0.962     5.383%
                                                          ∗∗∗
                    Substitution                     0.499      2.533    −1.561              0.967     5.227%
             Truncated Multiplication                0.733∗∗∗ −4.065∗∗∗    2.738∗∗∗          0.969     5.010%
                                                          ∗∗∗        ∗∗∗
           Multiplication and Substitution           0.658     −3.193      3.480∗∗∗          0.962     5.636%
                                                          ∗∗         ∗∗∗
           Normally Distributed Distance             0.954    −13.704    193.875             0.992     5.656%
 ***
       p < 0.01, ** p < 0.05, * p < 0.1

                         Table 3.2: The meanRT and stdRT Model on TSP Instances


                                Variable Coefficients
                                                        ln(stdRT )       constant          Adj.R2     MAPE
 Instance Set
                       Uniform                           1.552∗∗∗         2.091∗∗∗         0.881      14.265%
                        Arcsine                          1.642∗∗∗         1.584∗∗∗         0.921      12.079%
                        Normal                           1.717∗∗∗         3.503∗∗∗         0.950      10.840%
                    Multiplication                       0.873∗∗∗         2.198∗∗∗         0.380      25.827%
                     Substitution                        1.777∗∗∗         1.870∗∗∗         0.932       7.922%
              Truncated Multiplication                   1.458∗∗∗         2.092∗∗∗         0.633      22.222%
            Multiplication and Substitution              1.052∗∗∗         2.182∗∗∗         0.483      23.833%
            Normally Distributed Distance               −0.526∗∗∗        10.295∗∗∗         0.667      30.008%
 ***
       p < 0.01, ** p < 0.05, * p < 0.1

                                 Table 3.3: The stdRT Model on TSP Instances




3.3        Using mean and std to Predict VRP Tour Length

          The vehicle routing problem (VRP) is another extensively studied combinatorial optimiza-

tion problem. Unlike in the TSP, multiple vehicles are allowed to visit vertices, but each vertex

must be visited exactly once. In the VRP, vehicles have a fixed capacity and the objective is to

find the minimum total distance required to visit all vertices subject to the capacity constraint.

          To provide an overview of estimation models for the VRP, we include some models in Ta-

ble 3.4. In Daganzo’s model (1984), r stands for the average distance between the depot and

customers, N is the number of vertices, A is the area of the region in which the vertices are

                                                        34


--- PAGE BREAK ---

located, and C is the maximum number of customers a vehicle can serve. Robust, Daganzo, and

Souleyrette II (1990) provide a similar model. In this model, a1 is the area shape constant, and

it can be found through regression. Figliozzi (2008) includes an additional feature m, which is

the number of routes. In the third row of Table 3.9, a1 , b1 , and c1 represent coefficients obtained

from linear regression. More recently, Akkerman and Mes (2022) start with around 40 topo-

logical features and perform feature selection using simulated instances with different customer

distributions and demand patterns. They obtain a linear model that performs well on their waste

collection instances.
         Authors                        Predictive Model
                                                       √                         Experimental Instances
     Daganzo (1984)                  D = 2rN
                                           C
                                               + 0.57 √N A               Uniform customers in 10 × 10 square
    Robust et al. (1990)             D = [0.9 + aC1 N
                                                    2 ]  NA                Uniform customers in rectangles
                                       −m
                                          √           q                  Distinct spatial customer distributions,
     Figliozzi (2008)         D = a1 N N    N A + b1 N   A
                                                           + c1 m
                                                                                capacities, and demands
 Akkerman and Mes (2022)     Linear model with 18 topological features        Waste collection application

                        Table 3.4: Different Prediction Models in the Literature




      We observe from Table 3.4 that none of the four models has a simple format. Furthermore,

none of these models can predict the optimal tour length for a variety of CVRP instances well. In

this note, we will provide a model that is both simple and effective.

      Using the modified Clarke and Wright algorithm from Sinha Roy et al. (2023), we can

easily generate numerous feasible solutions to VRP instances. In contrast to our other experi-

ments, for which we find random feasible solutions, these feasible solutions are random heuristic

solutions. Instead of strictly considering the three best savings probabilistically as in Sinha Roy

et al. (2023), we consider either the three, four, or five best savings with equal probability when

generating feasible VRP solutions using the modified Clarke and Wright algorithm. In addition

and as in Sinha Roy et al. (2023), we apply Yellow’s parameter in the calculation of savings. Let

                                                     35


--- PAGE BREAK ---

meanCW i and stdCW i represent the mean and standard deviation in 1,000 distances output by the

modified Clarke and Wright algorithm when considering the best i savings, and then we can see

how the mean and standard deviation work in predicting the optimal distance for the VRP, which

is denoted by D.

       For the VRP, we consider models such as (3.6) and (3.7) for i = 3, 4, and 5 and use obtained

data to fit constant coefficients a3 , b3 , b4 , c3 and c4 .



                                D = a3 × meanCW i + b3 × stdCW i + c3                          (3.6)




                                      ln(D) = b4 × ln(stdCW i ) + c4 .                         (3.7)


       To understand the relationship between the mean and standard deviation of feasible heuris-

tic solution values and D, we consider instances from the VRP study by Queiroga, Sadykov,

Uchoa, and Vidal (2021). We use 162 instances from the set with names starting with XML 100 111.

These instances have a random uniform depot and customer positioning, unitary demands, and

average route sizes ranging from U [3, 5] to U [25, 50], and the optimal solutions to these instances

are given. For each instance, 1,000 solutions are obtained using the modified Clarke and Wright

heuristic with three best savings, another 1,000 solutions with four best savings, and another

1,000 with five best savings. To be specific, in the case of considering the best i savings, routes

are merged based on considering the largest i savings with equal probability 1i . Let C denote the

capacity of the vehicle, A denote the convex hull area, and N denote the number of vertices. We

compare the performance of different models in Table 3.5. In addition, we include some sum-



                                                       36


--- PAGE BREAK ---

mary statistics of the meanCW i and stdCW i predictors in Table 3.6 to help understand how the

number of savings under consideration can influence the predictive model.

                           Fitted Model                           Adj.R2      MAPE
            D = 1.013meanCW 3 − 0.742stdCW 3 − 1107.384           1.000        1.100%
            D = 1.013meanCW 4 − 0.854stdCW 4 − 1094.554           1.000        1.076%
            D = 1.012meanCW 5 − 0.972stdCW 5 -1076.734            1.000        1.054%
                ln(D) = −0.679 ln(stdCW 3 ) + 13.548              0.167       36.283%
                ln(D) = −0.813 ln(stdCW 4 ) + 14.342              0.207       35.749%
                ln(D) = −0.902 ln(stdCW√5 ) + 14.872              0.237       34.963%
                      D = [0.9 + 0.364N
                                   C2
                                        ] NA                       N/A        15.687%

                         Table 3.5: Different Models on VRP Instances



          Statistics    N        Mean           St. Dev         Min           Max
          meanCW 3     162    18865.493       9055.909       9022.304      56631.051
          meanCW 4     162    18896.302       9051.016       9070.492      56611.552
          meanCW 5     162    18927.513       9047.765       9155.715      56641.617
           stdCW 3     162      314.247         86.008        117.016        551.531
           stdCW 4     162      322.537         82.049        126.022        520.775
           stdCW 5     162      328.957         80.477        129.914        518.488

                     Table 3.6: Summary Statistics of meanCW i and stdCW i



      As we can observe from Table 3.5, models using both the meanCW i and stdCW i predictors

can predict the value of D reasonably well, while models using only the stdCW i predictor cannot

predict D well. Note that in Table 3.5, all predictors and constants except for the bold one are

significant at p = 0.01. In addition, the predictive results improve slightly as we increase the

number of best savings being considered. For comparison, the last row of Table 3.5 provides

the result of a more traditional model on predicting the optimal VRP distance (Akkerman et al.,

2020). The constant 0.364 in the formula is the fitted area shape constant, and since there is not a

constant term here, we do not have an adjusted r-squared value for the last model for comparison.

In addition, for this non-linear model, we do not include the significance level for independent

                                                37


--- PAGE BREAK ---

predictors. However, by comparing the MAPEs, we can observe that the meanCW i and stdCW i

model clearly outperforms the traditional model. We also note that generating 1,000 feasible

solutions for a VRP is not very time consuming. For reference, we code the modified Clarke and

Wright algorithm in Python, and for instances with size N = 100, generating 1,000 solutions

takes an average of 13.67 seconds on an M1 chip with 8 GB RAM.

      Since we observe good performance from our model, we extend this experiment to all

10,000 instances with known optimal solutions introduced by Queiroga et al. (2021). Addition-

ally, we note from Table 3.6 that considering more savings can, in general, result in worse VRP

heuristic solutions on average. The purpose of our experiment is not to solve the VRP instances

optimally or near-optimally. In fact, we identify that the summary statistics of meanCW i and

stdCW i do not vary greatly for i = 3, 4, and 5. Thus, for simplicity, in all experiments below we

consider the modified Clarke and Wright algorithm with the top three savings.

      The 10,000 instances introduced by Queiroga et al. have a two-dimensional Euclidean dis-

tance metric and the same number of customers, N = 100, and the depot and customers have

integer coordinates corresponding to points in a [0, 1000] × [0, 1000] grid. Note that the 162

instances from the previous experiment constitute a subset of these instances. These 10,000 in-

stances have either a random, centered, or cornered depot. The customer positioning is either

random, clustered, or half random and half clustered. There are also seven different types of

demands: unitary, small values with large variance, small values with small variance, large val-

ues with large variance, large values with small variance, variance depending on the quadrant,

or demand composed of many small values and a few large values. There are also six different

average route sizes ranging from very short to extremely long. Details of instance formulation

can be found in Queiroga et al. (2021). The key point here is that although instances in this

                                               38


--- PAGE BREAK ---

dataset share some similarities, they differ greatly in depot and customer distribution, demand,

and average route size.

      For this set of VRP instances, we test the performance of the models included in Table 3.5

as well as models including only meanCW 3 or stdCW 3 to study the influence of both predictors.

In addition to comparing these models with the one in the bottom row of Table 3.5, we add

another model considering the following list of features: number of customers N, capacity of

the vehicle C, perimeter of the convex hull P, area of the convex hull A, height of the enclosing

rectangle H, width of the enclosing rectangle W, average distance between customers dc , means

of variances in customer latitude and longitude meanV ar(x,y) , average distance between the depot

and all customers dd , average demand demand, and variance in demand V (demand). This list

of features comes from Akkerman et al. (2020). They built linear regression, random forest, and

neural network models using 38 features. After feature selection, 27 of the 38 were included in

their linear model. On their VRP instance set, the linear model attains an adjusted r-squared value

of 0.836 and a relative mean absolute error of 4.8%.

      We compare six models on the 10,000 VRP instances in Table 3.7. The last model is based

on the 11 features that are most important and easy to compute from Akkerman et al. (2020),

mentioned earlier. We obtain this model by stepwise regression using backward elimination.

Note that N = 100 for these 10,000 instances, so the feature N is not included in the model

after stepwise regression. From this table, we see that our meanCW 3 and stdCW 3 model clearly

outperforms the model in row 6 which is based on Akkerman et al. (2020).

      From Table 3.7, we clearly observe that the model containing both the meanCW 3 and the

stdCW 3 predictors outperforms the rest. From rows 1 to 4, we note the improvement in prediction

is mostly due to the additional meanCW 3 predictor, not the model structure. Also, the model in

                                                39


--- PAGE BREAK ---

                           Fitted Model                                 Adj.R2     MAPE
           D = 0.975meanCW 3 − 1.780stdCW 3 − 349.382                   0.999       1.808%
                 ln(D) = 0.337 ln(stdCW 3 ) + 7.934                     0.136      41.778%
                 D = 41.312stdCW 3 + 1.033 ∗ 104                        0.167      49.147%
                  D = 0.968meanCW 3 −√505.370                           0.998       1.942%
                     D = [0.9 + 0.406N
                                   C2
                                       ] NA                              N/A       38.083%
      D = −10.186C + 0.007A − 0.462P + 0.070W + 0.637H
     −9.084dc + 0.016meanVar(x,y) + 21.679dd + 147.258demand             0.395     40.525%
                  +2.979V (demand) + 3809.593

                        Table 3.7: Different Models on 10,000 VRP Instances



row 1 not only outperforms the model using only the stdCW 3 predictor but also the classic model

in row 5 and the model containing 10 instance features in the last row. Based on the models in

rows 1 and 4, one may question whether stdCW 3 is really helpful in predicting D. As in Table

3.5, we indicate non-significant predictors (at p = 0.01) in bold. In fact, for this instance set,

although the improvement is not obvious, the stdCW 3 predictor is significant, with p-value below

0.0001. In addition, in the next set of experiments, we observe that adding the stdCW 3 predictor

can indeed improve the predictive result.

      For 131 instances with known optimal solutions and size N > 50 from CVRPLIB (Uchoa

et al., 2017), we perform similar experiments. These instances come from 10 different papers

and have sizes ranging from N = 50 to 469; thus, the generation of instances was far from homo-

geneous, and vertex locations come from different distributions. We compare the performance

of the same models in Table 3.7 on instances in this set to verify that the meanCW 3 and stdCW 3

model is still valid.

      From Table 3.8, we observe that the model containing both the meanCW 3 and the stdCW 3

predictors outperforms the others, with an adjusted r-squared value close to 1 and an MAPE

around 2.5%. Again, based on the results in the first four rows, we see that both predictors are


                                                40


--- PAGE BREAK ---

                           Fitted Model                                        Adj.R2    MAPE
            D = 0.971meanCW 3 − 4.628stdCW 3 + 14.498                          1.000      2.541%
                 ln(D) = 1.084 ln(stdCW 3 ) + 4.645                            0.839     57.457%
                  D = 173.574stdCW 3 + 2534.214                                0.468    237.903%
                  D = 0.959meanCW 3 −√148.072                                  0.999      9.237%
                      D = [0.9 + 0.313N
                                   C2
                                        ] NA                                    N/A      42.166%
 D = 126.300N − 44.977C − 0.078A + 28.250P − 65.581W − 89.895H
             +87.761dc + 65.514dd + 438.801demand                               0.658   375.240%
                    +0.072V(demand) − 13140

                    Table 3.8: Different Models on 131 CVRPLIB Instances



essential when estimating D. In contrast to the results in Table 3.7, we note that including the

stdCW 3 predictor can greatly improve the performance of the model. To be specific, including

the stdCW 3 predictor decreases the MAPE by around 7% as compared to the model with only

meanCW 3 . Lastly, for this set of experiments, the model in row 1 outperforms the classic model

and the model with multiple instance features. As in Table 3.7, we indicate non-significant pre-

dictors (at p = 0.01) in bold. To conclude our experiment on VRP instances, although there is

a strong linear relationship between D and meanCW 3 , both predictors are essential in order to

build a model with a high adjusted r-squared value and a low MAPE. This precise approximation

is important for the VRP, for which fast and powerful heuristics such as LKH do not exist and for

which it is time-consuming to compute the optimal solution.

      To summarize our VRP experiments, we compare the six models in Table 3.7 and 3.8 on

the joint set of 10,000 VRP instances and 131 CVRPLIB instances. Note that this set includes

all VRP instances we experimented with, and we observe from Table 3.9 that the meanCW 3 and

stdCW 3 model still outperforms the remaining models on this diverse set, with a significantly

lower MAPE of 2.277%. Again, the model in the last row comes from backward stepwise re-

gression on all VRP instances by using the adjusted r-squared value as the criteria. Thus, the


                                               41


--- PAGE BREAK ---

predictors present differ slightly from the last row in Tables 3.7 and 3.8. In addition, all predic-

tors in Table 3.9 are significant at p = 0.01.

                            Fitted Model                          Adj.R2     MAPE
             D = 0.974meanCW 3 − 1.809stdCW 3 − 318.584           0.999       2.277%
                  ln(D) = 0.484 ln(stdCW 3 ) + 7.198              0.258      46.339%
                    D = 44.042stdCW 3 + 9890.980                  0.174      63.344%
                    D = 0.967meanCW 3 −√478.002                   0.998       2.632%
                      D = [0.9 + 0.390N
                                    C2
                                        ] NA                       N/A       38.191%
                 D = 151.133N − 9.295C + 0.005A
                     22.331dd + 125.492demand                     0.387      55.749%
                     +0.008V (demand) − 13860

                        Table 3.9: Different Models on all VRP Instances




3.4    Using mean, std, and min to Predict VRP Tour Length

      We consider three different instance sets as in the previous section: the 10,000 VRP in-

stances set by Queiroga et al. (2021), the CVRPLIB instances with size N > 50 (Uchoa et al.,

2017), and the joint set of the previous two instance sets. The predictive results of two different

models (with and without the min predictor) for the first dataset are included in Table 3.10.

      The second dataset contains 117 instances with known optimal solutions and size N > 50

from CVRPLIB (Uchoa et al., 2017). These instances come from nine different papers and have

sizes ranging from N = 50 to 469. As a result, the generation of instances is characterized by

significant heterogeneity, with vertex locations originating from various distributions.

      One thing to note is that these 117 instances come from the 131 CVRPLIB instances we

experimented with in the previous section (Kou, Golden, & Poikonen, 2024). Fourteen instances

in Christofides, Mingozzi, and Toth (1979) have been removed since they contain distance-

constrained as well as capacity-constrained VRPs. As a consequence, our MAPE results in the

                                                 42


--- PAGE BREAK ---

previous section are conservative. We compare our two models on the 117 CVRPLIB instances

in Table 3.11. It is worth pointing out that the MAPE in the first row of Table 3.11 is clearly

lower than the corresponding MAPE in the first row of Table 3.8 in the previous section (with

131 CVRPLIB instances).

      The points below describe how we generate the predictors.


    • For each instance, we randomly generate 1,000 modified Clarke and Wright solutions

      (Sinha Roy et al., 2023).


    • For these 1,000 solutions, we compute the mean, standard deviation, and minimum of their

      tour lengths.


    • Then, for the 10,000 VRP instance set, the 117 CVRPLIB instance set, and the joint set

      containing 10,117 instances, we consider the two linear regression models shown in Equa-

      tions (3.8) and (3.9) below and use the obtained predictor values to fit constant coefficients

      a1 , a2 , b1 , b2 , c2 , d1 , and d2 via linear regression:




                                 D = a1 × meanCW + b1 × stdCW + d1                             (3.8)


                      D = a2 × meanCW + b2 × stdCW + c2 × minCW + d2 .                         (3.9)


      The modified Clarke and Wright algorithm above starts with an initial solution that uses

one vehicle to visit each customer and a set of savings by merging any two short routes, as in the

standard Clarke and Wright algorithm. Then, in our implementation, at each iteration, routes are

merged based on considering the largest 3 savings with equal probability 31 .



                                                        43


--- PAGE BREAK ---

      In Equations (3.8) and (3.9), meanCW , stdCW , and minCW denote the mean, standard de-

viation, and minimum of the 1,000 modified Clarke and Wright solution tour lengths. D denotes

the optimal VRP tour lengths, which are available in the literature for all instances we exper-

imented with. We use 1,000 modified Clarke and Wright solutions since in Kou, Golden, and

Poikonen (2024), we ran some convergence tests and concluded that 1,000 solutions are easy to

generate and provide relatively stable mean, standard deviation, and minimum predictors.

      For the 10,000 VRP instances set in Table 3.10, note that all bold predictors and constants

are significant at p = 0.01. To compare the two models, we provide their adjusted r-squared

values and the mean absolute percentage error (MAPE) in the second and third columns, respec-

tively. From this table, we see that both models provide good predictions of the optimal VRP tour

lengths, and, in both models, all predictors are significant, indicating that all predictors contribute

to the predictive results. The adjusted r-squared values are 0.999 for both models, but we obtain

a 0.338% decrease in the MAPE when the minCW predictor is added.

      It is also worth noting that the coefficient for the stdCW predictor is negative in both models,

and this means the optimal VRP tour length is negatively correlated with the standard deviation in

the modified Clarke and Wright solution tour lengths. Meanwhile, the coefficient of the meanCW

predictor changes from positive to negative from the first to the second row. This may be because

when we do not include the minCW predictor, the meanCW predictor provides an estimate on

the magnitude of the optimal VRP tour length, and when the minCW predictor is included, it and

the meanCW predictor together determine the magnitude of the optimal VRP tour length, as the

coefficient for the minCW predictor is slightly greater than 1, and the coefficient for the meanCW

predictor is a small negative number.

      Bold predictors are significant at p = 0.01 in Table 3.11 as well. From this table for 117

                                                  44


--- PAGE BREAK ---

                       Fitted Model                                        Adj.R2      MAPE
          D = 0.979meanCW − 0.312stdCW −247.973                            0.999       1.808%
   D = −0.065meanCW − 0.213stdCW + 1.044minCW −240.928                     0.999       1.470%

                   Table 3.10: Different Models on the 10,000 VRP Instances


CVRPLIB instances, both the meanCW and stdCW predictors are significant in the first model,

but when the minCW predictor is added, it is not significant, and the meanCW predictor is also not

significant. This can be explained by the multicollinearity as shown in Figure 3.1. As these two

predictors are highly linearly dependent, we can get large errors when fitting their coefficients,

and this makes them insignificant in the second model. However, this validates our conjecture

that the minCW predictor and the meanCW predictor jointly determine the magnitude of the

optimal VRP tour length. On the other hand, by comparing the adjusted r-squared values and the

MAPEs in the first and the second rows, we observe that adding the minCW predictor improves

the MAPE by 0.162%.

                        Fitted Model                                     Adj.R2      MAPE
            D = 0.971meanCW − 4.580stdCW + 2.980                         0.999       2.155%
     D = 0.531meanCW − 3.570stdCW + 0.439minCW + 14.234                  0.999       1.993%

                 Table 3.11: Different Models on the 117 CVRPLIB Instances



      The predictive results of the two models for the joint set are presented in Table 3.12. Bold

predictors are significant at p = 0.01, and all predictors are significant in both models as shown.

Note that when the minCW predictor is added, the MAPE of the model decreases by around

0.4%, which is impressive as the model including the meanCW and the stdCW predictors already

has a low MAPE of about 2%.

      It is also worth noting that, as we mention in the introduction, little extra work is required

for the addition of the minCW predictor. To summarize our VRP experiments, we observe that in

                                                45


--- PAGE BREAK ---

        Figure 3.1: The Multicollinearity between the meanCW and minCW Predictors


Table 3.12 on the diverse set containing instances from 10 different papers, using the meanCW ,

stdCW , and minCW model, we can accurately predict the optimal VRP tour lengths with an error

of about 1.6%.

                       Fitted Model                                      Adj.R2      MAPE
         D = 0.973meanCW − 1.788stdCW + −326.819                         0.999       2.004%
   D = −0.055meanCW − 0.220stdCW + 1.032minCW −218.103                   0.999       1.602%

              Table 3.12: Different Models on the Joint Set with 10, 117 Instances




3.5   Conclusion

      This chapter extends previous research by demonstrating that random feasible solution

statistics can effectively estimate optimal solution values across a range of combinatorial opti-

mization problems, including the TSP, VRP, and other NP-hard routing problems. Our findings

validate and expand upon the power-law relationship between solution variance and optimal ob-

                                               46


--- PAGE BREAK ---

jective values, originally observed in the TSP, and confirm its applicability to other optimization

problems, including MST, MWM, and CVRP.

      For the TSP, we showed that incorporating both stdRT and meanRT improves predictive

accuracy, especially for non-Euclidean instances, where stdRT alone is insufficient. Similarly, for

the VRP, we introduced random heuristic-based predictors (meanCW and stdCW ), demonstrating

that these features significantly enhance tour length estimation. Further, by incorporating minCW

(the shortest heuristic solution found), we improved estimation accuracy while requiring minimal

additional computation.

      Our results confirm that mean and standard deviation of random feasible solutions provide

strong estimators for optimal objective values. For VRP, incorporating the minimum heuristic

tour length (minCW ) further refines predictions. These models generalize well across diverse

problem instances, including heterogeneous VRP datasets.

      The effectiveness of these randomization-based approaches suggests promising directions

for further research. In addition to refining estimation models, future work could explore their in-

tegration with heuristic and hybrid optimization algorithms, potentially leading to more efficient

routing heuristics with better performance guarantees.




                                                47


--- PAGE BREAK ---

Chapter 4:      Optimal TSP Tour Length Estimation Using Sammon Maps




4.1   Introduction

      Recall that there is a considerable amount of literature on statistical estimation of the opti-

mal tour length in the TSP. The most well-known and commonly used simple model was proposed

by Beardwood et al. (1959). For vertices identically, independently, and uniformly distributed on

a Lebesgue set with a positive measure in R2 ,


                                            L∗   √
                                        lim √ = β A,                                           (4.1)
                                       N →∞  N


where L∗ is the optimal TSP tour length, N is the number of vertices, β is an unknown constant,

and A is the measure of the Lebesgue set. We can think of A as the area including all vertices.

Based on this theorem, one can estimate L∗ using the formula


                                                √
                                          L∗ ≈ β N A.                                          (4.2)



Beardwood et al. (1959) approximated β to be about 0.75 when the pairwise distance is calculated

using the Euclidean (L2 ) metric. Chien (1992) argued that for small L2 instances with size 5 ≤

N ≤ 30, β ≈ 0.82. Daganzo (1984) derived upper bounds of β for the Manhattan distance (L1 )



                                                 48


--- PAGE BREAK ---

metric and the L2 metric. The value of β for the L1 metric was derived using the distance ratio

between the L1 and L2 metrics for a randomly distributed pair of vertices.

      Based on the idea of using the distance ratio between metrics, the circuity factor (also called

the directness), which is the average ratio of network to L2 distance, can be used to estimate

the optimal tour length when the metric is non-Euclidean. Merchán and Winkenbach (2019)

introduced the following model for estimating optimal TSP tour lengths in the context of urban

last-mile problems:
                                                 √
                                          L∗ ≈ cβ N A,                                         (4.3)


where β is still the constant for the L2 metric, and c is the circuity factor of the TSP instance

calculated using
                                           1       X d∗ (i, j)
                                     c=    N
                                                                   .                          (4.4)
                                           2     i̸=j∈V
                                                        d L2 (i, j)


In equation (4.4), d∗ (i, j) is the distance between vertex i and j in the network and dL2 (i, j)

is the L2 distance between vertices i and j. Instead of using a regression model, Merchán and

Winkenbach (2019) used fixed β = 0.9, which is the upper bound derived by Daganzo (1984).

For an urban last-mile TSP generated using a road map in São Paulo, Brazil, their model has an r-

squared value of 0.93 to 0.94 and mean squared percentage error (MAPE) of 8.72% to 12.65% for

three different sets of clustered instances based on dimensional and topological heterogeneities.

      In this chapter, we develop a way to estimate the optimal TSP tour length using the tech-

nique of Sammon mapping (Sammon, 1969) with only the distance matrix describing the pair-

wise distances among vertices in V . We argue that our model provides results comparable to

those of Merchán and Winkenbach (Merchán & Winkenbach, 2019), while also being applicable

to a wider variety of instances. We show this by comparing both models using regression and

                                                   49


--- PAGE BREAK ---

illustrating additional instances that our model can be applied to.



4.2   Estimation of Optimal Tour Length Using Sammon Maps

      Introduced by Sammon (1969) and related to the multidimensional scaling technique, Sam-

mon mapping aims to minimize the following error function:


                                         1        X (d∗ (i, j) − d(i, j))2
                            E= P                                           ,                (4.5)
                                        d∗ (i, j) i<j       d∗ (i, j)
                                  i<j



where d(i, j) is the L2 distance between the projection of vertices i and j and d∗ (i, j) is the

network distance between vertices i and j.

      We use Sammon mapping to find a two-dimensional coordinate representation that pre-

serves pairwise distances of vertices in V ; using these coordinates and the Convex Hull function

from Scipy (Virtanen et al., 2020), we then obtain the area of the convex hull over the projected

vertices. It is worth noting that the error function in Sammon maps pays more attention to the

shorter edges, and these edges are usually also the candidate edges in determining the optimal

TSP solution.

      Consider the following set of figures. Figure 4.1 is a piece of road map in São Paulo we

extracted using the OSMnx (Boeing, 2017) module. Note that we highlight two vertices in the

road map in red. Figure 4.2 shows the NetworkX (Hagberg, Swart, & S Chult, 2008) plot of

this road network, which can be considered as a generalized plot with exact locations but not

exact edge lengths. We can see that geographically, two red vertices are close to each other.

However, moving between these vertices requires traveling a long distance through the existing

edges. Figure 4.3 is a Sammon map generated using the pairwise distance matrix of the road map.

                                                   50


--- PAGE BREAK ---

We do not include edges in the Sammon map because the Sammon map is considered to be fully

connected, and the distance between vertices will be according to the L2 distance displayed in

the plot. On the Sammon map we obtain, we can see that the two red vertices are now relatively

far from each other.




                            Figure 4.1: A Road Map in São Paulo




                            Figure 4.2: Abstract of the Road Map




                                             51


--- PAGE BREAK ---

                             Figure 4.3: A Sammon Map of the Road Map



      Using the area of the convex hull in the Sammon map, we propose the following model to

estimate the optimal tour length for the TSP:


                                                    p
                                           L∗ ≈ β    N ASammon .                           (4.6)



Our experiments are designed as follows. Using the circuity factor, the area of the convex hull

based on the original coordinates, and the area of the convex hull in a Sammon map based on

projected coordinates, we use linear regression to fit the following models and find the numeric

values for the constants β1 , β2 , δ1 , and δ2 :


                                                 √
                                      L ∗ = β 1 c N A + δ1                                 (4.7)
                                                   p
                                      L ∗ = β2      N ASammon + δ2                         (4.8)



where A denotes the area of the convex hull based on original coordinates, ASammon denotes

the area of the convex hull based on projected coordinates, and the optimal tour length L∗ is

estimated using LKH, which is known to produce near-optimal solutions within 1% of optimality


                                                     52


--- PAGE BREAK ---

(Helsgaun, 2000). On each set of instances, we compare the two models’ performance.



4.3   Results on São Paulo Road Maps

      We first seek to duplicate the work of Merchán and Winkenbach (2019). According to
                             √
Merchán and Winkenbach the c N A model predicts L∗ well in last-mile delivery problems on
                                                     √
São Paulo road maps. We would like to test if the       N ASammon model also predict L∗ well in

our experiments. By selecting longitude and latitude uniformly as road map centers, we gen-

erate and proceed with 100 local maps of area 1 km2 in São Paulo using the OSMnx package

(Boeing, 2017). These 100 road maps are then considered to be 100 distinct asymmetric TSP in-

stances. For each instance, we can obtain pairwise distances, vertex locations, and circuity. Since

Sammon mapping requires symmetric distance matrices, we symmetrize the distance matrices by
                                                                     ∗      ∗
considering the new pairwise distance dnew (i, j) = dnew (j, i) = d (i,j)+d
                                                                         2
                                                                            (j,i)
                                                                                  . These instances

range in size from 12 to 297 vertices and include road maps from rural, urban, and intermediate

areas. Unlike Merchán and Winkenbach (2019), we use these 100 instances and divide them into

a training set of size 75 and a test set of size 25, and then we use the Scikit-learn (Pedregosa et

al., 2011) package to run linear regressions to test both models mentioned in Section 4.2. Table

4.1 shows the comparison. Since the training and test sets are split randomly, we do not observe

any overfitting throughout our experiments.

      We can observe from Table 4.1 that regression yields similar estimation results for both
              √
models. The       N ASammon predictor has an r-squared value of 0.849 on the test set, which is
                                        √
slightly higher than the 0.822 for the c N A predictor, but it is the other way around on the
                                                         √
training set. Also, the MAPE on the test set for the      N ASammon predictor is lower than the



                                                53


--- PAGE BREAK ---

              √
MAPE for the c N A predictor. Additionally, we note that our Table 4.1 experiments do not

separate groups of instances based on their dimensional and topological varieties and then form

a distinct model for each group, as was done in Merchán and Winkenbach (2019). Instead, we fit

a single set of model parameters over all instances, which, unsurprisingly, yields slightly lower

r-squared values than 0.93 − 0.94 in Merchán and Winkenbach, while the MAPE values are

comparable to the 8.7% − 12.7% range in Merchán and Winkenbach (2019).
                                                                              √           √
       It is a common question to ask whether using two separate predictors       N and    ASammon

or N and ASammon can predict L∗ better than the single predictor. We run experiments on

this set of instances, and it turns out that two combinations of variables do not outperform the
√                                                         √     √
    N ASammon model. In particular, the combination of     N and ASammon does not predict
                    √
L∗ as well as the       N ASammon predictor, and the combination of N and ASammon provides a
                                                                   √
slightly better MAPE value but a lower r-squared value than the        N ASammon model. Since the
√                                                                            √
    N ASammon predictor can also be viewed as a generalized version of the       N A predictor from

Beardwood et al. (1959) that is applicable to non-Euclidean instances, we stick with the model
            √
using the    N ASammon predictor.
                              √                            √
                  L∗ = 0.725c N A + 2.183       L∗ = 0.650 N ASammon + 1.878
                Training Set R-squared: 0.839    Training Set R-squared: 0.828
                 Training Set MAPE: 8.022%        Training Set MAPE: 8.708%
                  Test Set R-squared: 0.822        Test Set R-squared: 0.849
                   Test Set MAPE: 9.712%            Test Set MAPE: 8.283%




                                 Table 4.1: São Paulo Road Maps


                                                54


--- PAGE BREAK ---

      In practice, we are usually interested in estimating L∗ for larger instances. Thus, for 29 in-

stances from the previous experiment with size N > 200, we also randomly sample 100, 105, . . . , 195

vertices from each road map instance to generate new instances, and we compare the two models

with respect to these instances. This comparison of experimental results on these 580 (29 · 20)

instances is included in Table 4.2. For simplicity, instead of computing the circuity factor for

all newly generated instances, we use the circuity factor computed for the base instance that we

sample from.
                                 √                            √
                     L∗ = 0.727c N A + 2.455       L∗ = 0.709 N ASammon + 1.266
                   Training Set R-squared: 0.610    Training Set R-squared: 0.612
                    Training Set MAPE: 7.393%        Training Set MAPE: 7.642%
                     Test Set R-squared: 0.678        Test Set R-squared: 0.639
                      Test Set MAPE: 6.795%            Test Set MAPE: 7.540%




                               Table 4.2: Sampled São Paulo Road Maps


                                                             √
      For this sampled set of instances, the model with the c N A predictor yields a better result
                             √                                            √
than the model with the          N ASammon predictor. The model with the c N A predictor has a

slightly higher r-squared value and lower MAPE on the test set. However, in general, we can
                                             √                      √
observe strong linear relationships between c N A and L∗ and between N ASammon and L∗ .

      The relatively high r-squared values and low MAPE values in Table 4.1 confirm the validity
           √                 √
of using       N ASammon or c N A to estimate the optimal TSP tour length. In Table 4.2, due to the

large number of observations, both models exhibit lower r-squared values, but we can still observe


                                                   55


--- PAGE BREAK ---

strong linear trends in the plots. In addition, MAPE values are still low for both predictors.

       In fact, if we do not have enough information to run regression to obtain the optimal coef-

ficients in the models, we can consider choosing a β for L2 instances from the existing literature.

In fact, by fixing β = 0.75 from Beardwood et al. (1959) and taking the intercept as 0, the
√
    N ASammon predictor would yield an MAPE of 9.767% on the first set and 8.238% on the sec-
                                                                                √
ond set, which is still a good prediction. For comparison, using β = 0.75, the c N A predictor

would yield an MAPE of 15.482% on the first set and 16.121% on the second set.

       In addition to experimenting on 1 km2 São Paulo road maps, we also generate 100 instances

of 2 × 2 = 4 km2 road maps in São Paulo by uniformly selecting longitude and latitude as road
                                √
map centers to study if the         N ASammon predictor also predicts L∗ well on larger instances.

These new instances have size 158 ≤ N ≤ 1983 with N̄ = 780.5, and, as we can observe
                         √                      √
in Table 4.3, both the       N ASammon and the c N A models are working well in predicting the
                                                                                      √
optimal TSP tour length for these larger instances. We note that from Table 4.3 the       N ASammon

model has higher r-squared values and lower MAPE values on both the training and test sets than
     √
the c N A model. Although r-squared values are lower than for the 1 km2 road map instances,
                                                        √
we still observe a strong linear relationship between       N ASammon and L∗ .
                                 √                                                         √
       We can observe that the    N ASammon predictor exhibits comparable accuracy to the c N A

predictor in the above experiments. However, it is worth noting that the computation of the circu-

ity factor and A requires known x-y coordinates in addition to the pairwise distances. On the other
                                                                                      √
hand, a Sammon map can be generated using pairwise distances only. Thus, the              N ASammon

model can be applied to more general TSP instances in which x-y coordinates are unknown.




                                                  56


--- PAGE BREAK ---

                             √                               √
                 L∗ = 0.478c N A + 17.031         L∗ = 0.474 N ASammon + 12.793
                Training Set R-squared: 0.677       Training Set R-squared: 0.744
                Training Set MAPE: 10.282%           Training Set MAPE: 9.847%
                  Test Set R-squared: 0.684           Test Set R-squared: 0.789
                  Test Set MAPE: 12.800%              Test Set MAPE: 11.044%




                         Table 4.3: Sampled 4 km2 São Paulo Road Maps




4.4      Results on Random TSPs

        We also run experiments on completely random instances to test the predictive ability of
      √
the    N ASammon predictor. We consider d(i, j) independent and uniformly distributed in (0, 100)

for all pairs of distinct vertices (i, j), and we repeat this process to generate 10 instances of each

size 20, 30, ..., 90. Then we add either 0, 5, 10, 20, 50, or 100 to all pairwise distances. We

expect that as we add more to the distances, there will be fewer triangle inequality violations.

Table 4.4 shows the r-squared value and MAPE for each regression model when different values

are added to edge length. For instances with a fixed added value, we fit a linear model using the
√
    N ASammon predictor for all instances with different sizes. In other words, each row in Table

4.4 is based on 80 instances.

        We observe from Table 4.4 that as the amount added to each edge length increases, using the
√
    N ASammon predictor yields better prediction results in terms of r-squared values. In additon,
                                                                                       √
even when the edge lengths are completely random, in the first row, fitting the            N ASammon



                                                 57


--- PAGE BREAK ---

 Added Value               Fitted Model                Prediction Results         Plots
                            √                          R-squared: 0.927
       0          L∗ = 1.367 N ASammon − 4.599




                                                        MAPE: 5.18%
                           √                           R-squared: 0.970
       5         L∗ = 2.135 N ASammon − 223.331




                                                        MAPE: 4.44%
                           √                           R-squared: 0.982
      10         L∗ = 2.700 N ASammon − 414.283




                                                        MAPE: 3.45%
                           √                           R-squared: 0.982
      20         L∗ = 3.673 N ASammon − 843.982




                                                        MAPE: 4.45%
                          √                            R-squared: 0.985
      50        L∗ = 5.352 N ASammon − 2025.496




                                                        MAPE: 4.93%
                          √                            R-squared: 0.988
     100        L∗ = 6.722 N ASammon − 3949.286




                                                         MAPE: 5.04%
                                     √
                        Table 4.4:    N ASammon on Random Instances



predictor still shows a relatively high r-squared value of 0.927 and low MAPE of 5.18%. The

promising results for models in each row also suggest that the same fitted constants β2 and δ2



                                              58


--- PAGE BREAK ---

                           Added Value   Average E        Triangle Inequality Rate
                                0          0.111                  75.83%
                                5          0.099                  81.48%
                               10          0.091                  86.63%
                               20          0.082                  93.15%
                               50          0.075                  99.56%
                              100          0.074                 100.00%

                               Table 4.5: Properties of Random Instances



work well for instances with sizes from 20 to 90. This indicates a strong linear relationship
                     √                               √
between L∗ and           N ASammon . Note that the    N ASammon predictor exhibits good prediction
                      √
results, whereas the c N A predictor is not applicable due to the lack of x-y coordinates.

          We include two statistics for instance sets with different added values in Table 4.5. Av-

erage E is the average error term in the Sammon maps for instances in a set, and the triangle

inequality rate measures the average likelihood that the triangle inequality is satisfied by a ran-

dom trio of vertices. While the error of a Sammon map for a perfect Euclidean instance is 0,

we can consider the average error as a rough measure of how far a TSP instance deviates from a

Euclidean TSP instance. Similarly, instances with higher triangle inequality rates can be viewed

as more structured and closer to Euclidean instances. Combining the results in Tables 1 and 2,

we conclude that when TSP instances are closer to Euclidean and more highly structured, using
      √
the       N ASammon predictor is expected to provide better results.



4.5       Results on Instances with Obstacles

          Recognizing that there are usually obstacles in road maps, such as parks, lakes, universi-

ties, and hospitals, we want to simulate this reality using some simple and randomly generated

instances. We start by considering a disk with radius 5, then a disk that shares the same center


                                                     59


--- PAGE BREAK ---

with radius 1, ..., or 4 is removed from the original disk. On these 4 rings, 100 instances with size

100, 101, ..., 199 and uniform vertex location distributions are generated.

          To avoid traveling through the removed central disk, each vertex is considered to be directly

connected to its 5 nearest neighbors, and then a shortest path algorithm from Scipy (Virtanen et

al., 2020) is run to complete the distance matrix. Considering 4 sets of instances with different

central areas removed, we compare models in Section 4.2 on them. In Tables 4.6 to 4.9, we

include the predictive results of both models instances on the disk with radius-1 disk removed,

with radius-2 disk removed, with radius-3 disk removed, and with radius-4 disk removed. We

observe from Tables 4.6 to 4.9 that, for these instances on rings, the predictive results on L∗ using
      √                      √
the       N ASammon and the c N A predictors are very similar.
                               √                             √
                  L∗ = 1.179c N A − 16.479 L∗ = 1.147 N ASammon − 9.198
                 Training Set R-squared: 0.870    Training Set R-squared: 0.848
                  Training Set MAPE: 3.690%        Training Set MAPE: 3.898%
                   Test Set R-squared: 0.819         Test Set R-squared: 0.883
                    Test Set MAPE: 4.137%             Test Set MAPE: 3.547%




                                   Table 4.6: Radius-1 disk removed




4.6       Results on 3DTSPs

                                                                                         √
          Inspired by the work of Yang et al. (2022), we find that our model using           N ASammon

can also be applied to TSPs in three-dimensional space (3DTSPs). Four sets of 3DTSPs are


                                                    60


--- PAGE BREAK ---

                               √                                  √
                  L∗ = 1.051c N A − 9.273              L∗ = 1.053 N ASammon − 5.785
                 Training Set R-squared: 0.878          Training Set R-squared: 0.871
                 Training Set MAPE: 11.890%              Training Set MAPE: 3.558%
                   Test Set R-squared: 0.893              Test Set R-squared: 0.883
                    Test Set MAPE: 2.731%                  Test Set MAPE: 2.340%




                                    Table 4.7: Radius-2 disk removed

                              √                                  √
                  L∗ = 0.829c N A + 2.475             L∗ = 0.760 N ASammon + 11.950
                Training Set R-squared: 0.776           Training Set R-squared: 0.771
                 Training Set MAPE: 4.132%               Training Set MAPE: 3.940%
                  Test Set R-squared: 0.801               Test Set R-squared: 0.736
                   Test Set MAPE: 3.783%                   Test Set MAPE: 4.604%




                                    Table 4.8: Radius-3 disk removed



constructed for the experiments. The first set comprises Euclidean 3DTSPs based on the work

of Yang et al. (2022), in which all instances have uniformly distributed vertices. All vertices

are distributed in cuboids with volume vol ∈ {1000, 8000, 27000, 64000, 125000} and ( wl , wh ) ∈

{(4, 1), (4, 2), (4, 3), (4, 4), (3, 1), (3, 2), (3, 3), (2, 1), (2, 2), (1, 1)}, where l, w, h denote the length,

width, and height of the cuboid, respectively. Instance sizes range from 10, 20, ..., 100. Consid-

ering all combinations, we construct 500 (5 · 10 · 10) instances. The second set of 500 3DTSPs



                                                      61


--- PAGE BREAK ---

                             √                               √
                 L∗ = 0.672c N A − 3.033          L∗ = 0.670 N ASammon − 1.284
               Training Set R-squared: 0.878       Training Set R-squared: 0.861
                Training Set MAPE: 3.171%           Training Set MAPE: 3.162%
                 Test Set R-squared: 0.841           Test Set R-squared: 0.865
                  Test Set MAPE: 3.211%               Test Set MAPE: 3.552%




                                Table 4.9: Radius-4 disk removed



               use
that we generate  thesame settingexcept
                                           that the vertex locations have a multivariate normal
                   l
                2  10 0 0 
                                  
                                  
distribution N 
                2  ,  0 10 0  instead of a uniform distribution, and we continue to
                 w                 
                                  
                                  
                   h
                   2
                          0     0   10
use Euclidean distance in this set. The third set is based on the idea of removing a cuboid within

the cuboid, which is inspired by our experiments with 2-dimensional instances with obstacles.

Considering the same sets of vol, wl , and wh , we remove a cuboid at the center of each cuboid with

dimension 2l × w2 × h2 . Then 10, 20, ..., 100 vertices are uniformly distributed in the remaining

space. The third set uses the L1 distance metric.
                                                                            √
      For these three sets of 500 3DTSPs, we test the performance of our        N ASammon predictor

as well as include two models below, which were proposed in Yang et al. (2022). Instances

are randomly split into a training set and a test set with ratio 0.75 to 0.25. For the purpose of

comparison, we use linear regression on the training sets to obtain the values of constants β3 , β4 ,




                                                 62


--- PAGE BREAK ---

δ3 , and δ4 . Results on both training and test sets are reported in the Tables 4.10 to 4.12 below.


                                            √      1
                                     L∗ = β3 N vol 3 + δ3                                        (4.9)
                                                         1
                                     L∗ = β4 (N vol) 3 + δ4                                    (4.10)




            √                                        √      1                                   1
 L∗ = 1.241 N ASammon + 12.613           L∗ = 1.685 N vol 3 − 12.164       L∗ = 1.687(N vol) 3 − 12.866
   Training Set R-squared: 0.982         Training Set R-squared: 0.988     Training Set R-squared: 0.938
    Training Set MAPE: 6.631%             Training Set MAPE: 6.172%        Training Set MAPE: 15.534%
     Test Set R-squared: 0.980             Test Set R-squared: 0.987         Test Set R-squared: 0.953
      Test Set MAPE: 6.570%                 Test Set MAPE: 6.129%            Test Set MAPE: 14.019%




                                   Table 4.10: Uniform 3DTSPs



            √                                        √      1                                   1
 L∗ = 1.223 N ASammon + 16.137           L∗ = [1.691 N vol 3 − 13.683       L∗ = 3.521(N vol) 3 − 38.407
   Training Set R-squared: 0.982         Training Set R-squared: 0.990      Training Set R-squared: 0.945
    Training Set MAPE: 6.522%             Training Set MAPE: 6.550%         Training Set MAPE: 15.412%
     Test Set R-squared: 0.980             Test Set R-squared: 0.984          Test Set R-squared: 0.931
      Test Set MAPE: 6.580%                 Test Set MAPE: 6.742%             Test Set MAPE: 15.954%




                                   Table 4.11: Gaussian 3DTSPs




                                                 63


--- PAGE BREAK ---

            √                                        √      1                                   1
 L∗ = 1.195 N ASammon + 19.671           L∗ = 2.569 N vol 3 − 16.326      L∗ = 5.414(N vol) 3 − 62.650
   Training Set R-squared: 0.990         Training Set R-squared: 0.975    Training Set R-squared: 0.935
    Training Set MAPE: 5.617%             Training Set MAPE: 7.890%       Training Set MAPE: 15.938%
     Test Set R-squared: 0.991             Test Set R-squared: 0.980        Test Set R-squared: 0.936
      Test Set MAPE: 5.189%                 Test Set MAPE: 6.458%           Test Set MAPE: 14.000%




                   Table 4.12: Uniform 3DTSPs with Central Cuboid Removed



       Table 4.10 contains the results on the first set of uniform 3DTSPs, and Table 4.11 contains

the results on the second set of 3DTSPs with multivariate normally distributed vertices. Table

4.12 concludes our experimental results on instances with the central cuboid removed. We ob-
                                                                     √
serve from Tables 4.10 and 4.11 that, on Euclidean instances, the        N ASammon model and the
                   √      1
model using the     N vol 3 predictor (Yang et al., 2022) both have relatively good performance
                                                 1
and outperform the model using the (N vol) 3 predictor. However, for instances in the third set,
                                √                                                          √         1
which are non-Euclidean, the        N ASammon model predicts L∗ slightly better than the       N vol 3

predictor, with slightly higher r-squared values and around 1% lower MAPEs. Yang et al. (Yang

et al., 2022) also proposed four more complicated models containing multiple predictors, and we

expect those models to outperform the simple models. Based on our experiments, this is true for
                                            √
the first and second 3DTSP sets, but the        N ASammon model outperforms both of the two sim-

ple models and the more complicated models on the third set. Thus, for 3DTSPs, we recognize
√
    N ASammon as a robust predictor with a relatively simple form.




                                                     64


--- PAGE BREAK ---

4.7   Further Thoughts and Conclusion

      The TSP is a combinatorial optimization problem that is expensive to solve optimally.

Thus, many researchers have studied how to accurately estimate the optimal tour length. In this
                                                              √
chapter, we propose a new method for estimation using the      N ASammon predictor. Given the

existence of heuristic solvers such as the Lin-Kernighan-Helsgaun Heuristic (LKH) (Helsgaun,

2000), it is reasonable to ask: Why is optimal TSP tour length estimation still important? We

believe there are a number of reasons.

      First, there is an academic argument. There is a long history of research on this topic,

starting with Beardwood et al. (1959) results from 1959 and continuing through the Yang et

al. (2022) results of 2022, described in this article. The characterizations in these papers yield

insights with respect to the problem structure, which may later be used directly or they may

provoke thought.

      Next, there is a computation time argument. The Sammon mapping method we employ

is O(N 2 ) in the worst case, and some recent papers have looked at mechanisms to speed up the

mapping further (Pekalska, de Ridder, Duin, & Kraaijveld, 1999; Wang, Fang, & Wang, 2016).

Though there is not a closed-form, worst-case computational complexity for LKH, empirically

the average run-time growth has generally been greater than O(N 2 ) for moderate to large TSPs

(Helsgaun, 2000). Though we did not focus on the vehicle routing problem (VRP) in this chapter,

a natural VRP extension to our Sammon mapping method may provide even stronger computation

time advantages relative to LKH-3 (Helsgaun, 2000) or alternative VRP heuristic solvers. We are

currently working on this extension.

      Thirdly, we think there is practical argument. A heuristic solution given by LKH or similar

                                               65


--- PAGE BREAK ---

is applicable to a particular instance with a prespecified set of nodes. By relying only on two

parameters, the number of nodes, N , and the Sammon map area, ASammon , the Sammon map

estimation is more readily used in settings where there may exist some uncertainty. For example,

suppose a company will deliver to some subset of customers in a given region. The subset of

customers may not be known fully in advance, and the company would like to estimate the opti-

mal TSP objective value considering this uncertainty. If they know that the Sammon map area is

approximately constant over their region and have a forecast model that gives the probability dis-

tribution of N , the number of customers, then an expected value of the optimal TSP tour length

is a straightforward computation.

        By running regression on the dataset generated from São Paulo road map instances and
                                                                 √
randomly generated instances with obstacles, we compare the          N ASammon predictor with the
 √
c N A predictor from Merchán and Winkenbach (2019). The regression results obtained in this

article show that both models provide similar estimation outcomes. In addition, it is worth noting

that even without building a regression model, we can use the theoretical β from the literature and
      √
the    N ASammon predictor to build a model that nicely estimates the optimal tour length in the

TSP for São Paulo road map instances. Moreover, the MAPE of the estimation results is almost

always less than 10%, which is reasonably accurate.
                                                √
        In fact, unlike the computation of the c N A predictor, which requires knowledge about the

coordinates of vertices, the proposed model can be applied to more general TSP instances where

the coordinates are unknown. For example, single machine scheduling can be formulated as a

TSP, but there are no meaningful coordinates (Laporte & Palekar, 2002). In addition to its routing

problems, the TSP may also be applied to cutting stock problems, genome sequencing, and the

structure of crystals (Hoffman, Padberg, Rinaldi, et al., 2013). For all these applications, it is

                                                66


--- PAGE BREAK ---

essential to have an estimation method that does not require knowledge of the vertex locations.
                   √
In addition, the       N ASammon predictor can be applied to 3DTSP instances and provide tour length

estimations comparable in quality to the simple models in Yang et al. (2022).




                                                   67


--- PAGE BREAK ---

Chapter 5:      Estimating Optimal Split Delivery Vehicle Routing Problem Solu-

                tion Values




5.1    Introduction

      In the classic Vehicle Routing Problem (VRP), we have vehicles, each with a certain ca-

pacity, tasked with supplying a group of customers whose demands are known in advance. A

constraint is that each customer must be visited by exactly one vehicle and the objective is to

minimize the overall distance that all vehicles travel. However, in the Split Delivery Vehicle

Routing Problem (SDVRP), the above constraint is relaxed, allowing the possibility that a cus-

tomer can be visited more than once to meet his demand, a practice known as split deliveries (see

Archetti and Speranza (2008)).

      In Figure 5.1, we illustrate the benefit that logistics providers can gain from allowing split

deliveries. Consider a scenario where vehicles have a fixed capacity of 100 and customers 1, 2,

and 3 have demands of 60, 80, and 60 units, respectively. In the classic VRP model, three separate

vehicles are required to meet the demands of customers 1, 2, and 3, since the combined demands

of any two customers exceed the vehicle capacity. The optimal solution in this scenario consists

of three routes: {[0, 1, 0], [0, 2, 0], [0, 3, 0]}. This arrangement results in a total travel length of

600. Conversely, in the SDVRP model, we can fully utilize the vehicle capacity of 100 by letting



                                                  68


--- PAGE BREAK ---

         Figure 5.1: Example Instance for VRP and SDVRP with demands D1, D2, and D3


two vehicles fulfill the 60-unit demands at customers 1 and 3, while the remaining capacity

is allocated to meet the needs of customer 2. An optimal solution for the SDVRP could be

{[0, 1, 2, 0], [0, 3, 2, 0]}, achieving a total travel length of 400 + 2ϵ, substantially shorter than the

optimal VRP length. In fact, in this instance the cost of the VRP tends to be 50% more than

the cost of the SDVRP, when ϵ tends to 0. This example clearly demonstrates that allowing split

deliveries can lead to a reduction in both the number of vehicles deployed and the total distance

traveled. In terms of distance, it is well known that the cost of the VRP is at most 100% more than

the cost of the SDVRP when the triangle inequality holds (Archetti, Savelsbergh, & Speranza,

2006).

      In this chapter, rather than seeking an exact solution, we employ a regression model to

estimate the optimal solution value for the SDVRP. This model provides multiple benefits. For

instance, the estimated tour lengths can serve as benchmarks, providing a standard for evaluating

the effectiveness of heuristic algorithms on new instances. Furthermore, logistics companies



                                                   69


--- PAGE BREAK ---

can use these estimates in operational planning to gauge potential savings from split deliveries;

this may help them decide whether to invest in the development of efficient SDVRP solutions.

Additionally, during the strategic planning stage, detailed operational specifics are not crucial.

For instance, making decisions about the placement of depots or determining the size of service

regions can be made independently of knowing the exact routes that will be traversed on any given

day. The availability of these estimates could be critical in improving the efficiency of solution

algorithms. For example, they can be used in Approximate Dynamic Programming (ADP) to

efficiently compute an estimate of the future cost for each state at each stage. In fact, since in

these ADP algorithms only the cost of the solution is needed, there is no need to spend additional

computational time to compute near-optimal solutions to estimate the future cost.

      The subsequent sections of the chapter delve into the details of our approach. A compre-

hensive literature review on the application of regression models to the VRP, as well as existing

approaches to the SDVRP, is presented in Section 5.2. In Section 5.3, we detail the heuristic

method we devised to derive estimators for our model, building upon insights from our prior re-

search on the Traveling Salesman Problem (TSP) and the VRP (Kou et al., 2022; Kou, Golden,

& Poikonen, 2024). Section 5.4 describes the empirical results of our model and its compara-

tive analysis with existing approximation models for the VRP. In Section 5.5, we illustrate an

application of the regression model, demonstrating its effectiveness in enhancing a heuristic al-

gorithm for the VRP. We conclude with Section 5.6, where we highlight our findings and outline

directions for future research.




                                               70


--- PAGE BREAK ---

5.2      Literature Review

         The Traveling Salesman Problem (TSP) is, perhaps, the most well-known problem in op-

erations research. Beardwood et al. (1959) proposed one of the earliest approximation models

for the optimal TSP tour length in Euclidean space using the number of vertices, n, and the area

containing all vertices, A. Later on, the estimation of the optimal TSP tour length was studied

by multiple researchers who improved upon the Beardwood et al. model, which contains the
            √
predictor       nA, and incorporated other topological features.

         For the capacitated VRP, which can employ multiple vehicles, Webb (1968)’s research laid

the groundwork by examining the relationship between the lengths of routes and the distances

from customers to the depot. Eilon, Watson-Gandy, Christofides, and de Neufville (1974) intro-

duced methods to approximate VRP route lengths taking into account various factors including

topological features, distance between customers and the depot, and vehicle capacities.

         Daganzo (1984) introduced an effective approximation model to estimate the optimal length

of a set of routes with random locations when all customers have unit demand:


                                                           √
                                  CV RP (n) ≈ 2r̄n/Q + 0.57 nA.                              (5.1)



         Here, CV RP (n) denotes the aggregate distance of routes serving n customers, r̄ denotes

the mean distance between customers and the depot, and Q denotes the maximum number of

customers that can be served per vehicle (Figliozzi, 2008). Therefore, n/Q calculates the smallest

number of routes required, making 2r̄n/Q a plausible lower bound on the VRP’s optimal solution

value.


                                                   71


--- PAGE BREAK ---

      Expanding on approximation models for the TSP, Figliozzi (2008) formulated six mod-

els for the VRP. These models approximate the optimal solution value based on the quantity of

customers served and the number of routes needed. Figliozzi tested the models empirically on

various instance types, characterized by distinct customer spatial arrangements, time windows,

demands, and depot positions. Among the six models, the one with the highest estimation accu-

racy is represented by equation (5.2). In this equation, b1 , b2 , and b3 are determined through linear

regression and are dependent on the characteristics of the VRP instances. Additionally, m = Qn

is the number of necessary routes; see the equation below:


                                         n − m√
                                                             r
                                                                 A
                           V RP (n) ≈ b1        nA + b2            + b3 m.                       (5.2)
                                           n                     n


      In 2019, Nicola et al. introduced regression-based estimation models for the TSP, the CVRP

with time windows, and the multi-region, multi-depot pickup and delivery problem. Starting with

multiple features such as distances, time windows, capacities, and demands, they applied step-

wise regression to distill these features into a simpler model. The derived regression models

generally provided close approximations to the total distance for both randomly generated exam-

ples and established instances from the literature.

      While these approaches yielded promising results for approximating route lengths, they

typically rely on the assumption of Euclidean distances between vertices. In contrast, Basel

and Willemain (2001) ran a regression on 17 Euclidean TSP instances from the TSPLIB repos-

itory (Reinelt, 1991), proposing an optimal tour length estimation model that uses the standard

deviation of random tour lengths as a predictor. Contrary to certain topological features, the stan-

dard deviation does not require the instance to be Euclidean (Kou et al., 2022). We proved the


                                                  72


--- PAGE BREAK ---

                                                                              √
asymptotic linear relationship of the standard deviation predictor with the    nA predictor from

Beardwood et al. and extended the standard deviation model to non-Euclidean TSP instances.

Further, we found that incorporating the mean of random tour lengths or random total lengths as

an additional predictor significantly improves the accuracy of the TSP and VRP optimal solution

estimates (Kou, Golden, & Poikonen, 2024). In Chapter 3 of the thesis, we included that for the

VRP, the inclusion of the minimum random solution value further enhances the precision of the

solution value approximation (Kou, Golden, & Bertazzi, 2024b).

      As outlined in the introduction, the Split Delivery Vehicle Routing Problem (SDVRP) is

an extension of the VRP, where the key distinction is the allowance of split deliveries. The

SDVRP was initially introduced by Dror and Trudeau (1989), and they showed that allowing

split deliveries can lead to significant cost savings. Archetti and Speranza (2008) provided a

comprehensive survey on the SDVRP, covering analytical studies, exact approaches, and heuristic

algorithms. In their review, they noted that due to the SDVRP’s complexity, exact algorithms were

constrained to instances with no more than 100 customers. This complexity comes from allowing

a single customer’s demand to be met through multiple routes, which amplifies the challenge for

exact algorithms. The computational time and resources required to explore the entire solution

space and guarantee optimality are typically impractical, especially for large-scale problems.

More recently, Bertazzi and Wang (2022) introduced matheuristic approaches to the SDVRP and

examined their worst-case performance. Given SDVRP’s complexity and broader solution space

compared to the VRP for an equivalent customer set, we aim to expand our prior research and

establish valid regression models for the problem. As discussed earlier, these models are not only

helpful in evaluating heuristic algorithms but can also play a vital role in the strategic planning

process of logistics providers and in the design of solution algorithms.

                                                73


--- PAGE BREAK ---

5.3   Modified Clarke & Wright Algorithm with Split Deliveries

      Generating random feasible solutions for the SDVRP is more complex than for the TSP,

where random solutions can simply be created by shuffling vertices. In Kou, Golden, and Poiko-

nen (2024), we adapted the modified Clarke & Wright algorithm, as detailed by Sinha Roy et al.

(2023), to generate random VRP solutions. This modified algorithm considers the top k savings,

with k > 1, with a specific probability distribution. This algorithm enables us to explore the

solution space and generate multiple different solutions quickly.

Algorithm 1 The Modified Clarke & Wright Algorithm with Split Deliveries (MCWSD)

   1. Compute savings for all pairs of customers (i, j), assign
      S(i, j) ← d(0, i) + d(0, j) − d(i, j), and build the initial set of routes [0, i, 0] for all cus-
      tomers i, where 0 denotes the depot.

   2. Sort the savings list from largest to smallest.

   3. Choose the top 3 savings with equal probability, and remove the chosen one from the
      savings list.

   4. Suppose that S(i, j) is chosen.

         • Search the current routes to look for a route of the form [0, i, v1 , ..., vm , 0] and another
           route of the form [0, u1 , ...un , j, 0]. For each of these routes, if it includes an i or j
           which has been split in two, choose the route with the smaller total demand. If we
           cannot find two such routes, go to Step 5.
         • If the total demand of two routes does not exceed the capacity, merge the routes into
           [0, u1 , ...un , j, i, v1 , ..., vm , 0] and record the total demand.
         • If the sum of demand exceeds the capacity, then we try to drop the excess demand
           at the customer that is closest to the depot from the list of {i, j, v1 , ..., vm , u1 , ..., un }.
           If this cannot satisfy the capacity constraint, we try to drop the excess demand at the
           customer that is second closest to the depot.
         • If we cannot merge the two routes, go to Step 5.

   5. Stop if the savings list is empty, else go back to Step 3.



      For the SDVRP, we employ a similar approach but with an extension for split deliveries, as

                                                    74


--- PAGE BREAK ---

described in our Modified Clarke & Wright for Split Deliveries (MCWSD) algorithm. Following

the methodology of Sinha Roy et al. (2023), the algorithm merges routes based on the savings,

selected with equal probability from the highest three. Detailed in Algorithm 1, it adopts the

Clarke & Wright savings algorithm’s strategy of working through savings from highest to lowest

for potential route merges (Clarke & Wright, 1964).

      While the classic Clarke & Wright algorithm only combines routes when the capacity con-

straint is not violated, we attempt to merge two routes if we can re-satisfy the capacity constraint

by splitting or dropping one customer. Note that when searching, we consider all current routes

including customers i and j as in Step 4 of Algorithm 1, but we do not modify routes when the

total demand already equals the capacity. This means that if customer i was split earlier and two

routes contain it, we will consider merging the route with lower total demand.

      When merging two routes, they can be combined as per the Clarke & Wright algorithm if

their combined demands do not exceed vehicle capacity. Otherwise, we proceed as described in

Step 4. Assume customer v1 is the closest customer to the depot among {i, j, v1 , ..., vm , u1 , ..., un }

and its demand is 10 units in the route [0, i, v1 , ..., vm , 0] as depicted in Figure 5.2. If the vehicle

capacity is 105 units and the combined demand is 110 units, after merging and applying a split

strategy, we form two new routes: [0, u1 , . . . , un , j, i, v1 , . . . , vm , 0] with a total demand of 105

units and [0, v1 , 0] with 5 units, as shown in a) and b) in Figure 5.2. Conversely, with a capacity

of 100 units as shown in c) and d) in Figure 5.2, customer v1 would be dropped to form routes

[0, u1 , . . . , un , j, i, v2 , . . . , vm , 0] with a demand of 100 units and [0, v1 , 0] with a demand of 10

units. If the capacity is further reduced to 95 units and the constraint cannot be met by splitting at

v1 , we then consider splitting at the second-closest customer to the depot, which is u1 in Figure

5.2. If all these measures fail, the routes are not merged, and we proceed to the next potential

                                                      75


--- PAGE BREAK ---

                     Figure 5.2: Merging two routes according to MCWSD


savings.



5.4   Regression Models and Experiment Results

      Using the MCWSD algorithm, we were able to rapidly generate multiple solutions for

SDVRP instances. Specifically, for the 95 instances ranging from 9 to 289 vertices available

on the DIMACS website, we have generated 1,000 solutions for each instance. This number of

solutions has been selected based on Kou, Golden, and Poikonen (2024), where it was determined

that 1,000 solutions achieve a balance between ease of generation and stability of mean and

standard deviation values as predictive features.

                                                76


--- PAGE BREAK ---

     Note that, in the SDVRP, each customer i with demand D(i) can be considered as D(i)

individual customers, each with unit demand, but sharing the same location. Consequently, an
                                                                 P
SDVRP instance can be transformed into a VRP instance with           i∈V D(i) customers. To assess


the accuracy of the regression models presented in equations (5.3), (5.4), and (5.5), we compare

them with three established models for the VRP in Section 5.2. These three models include the

CVRP model by Daganzo (1984) described in equation (5.1), the VRP model by Figliozzi (2008)

shown in equation (5.2), and the model introduced by Nicola et al. (2019) for the capacitated

vehicle routing problem with time-windows (CVRPTW). For the three VRP models, we denote
                                      P
the number of customers as N =        i∈V D(i). In equation (5.1), r̄ becomes the weighted average


distance between customers and the depot, with customer demand serving as the weight. Since

N/Q is equivalent to m in equation (5.2), we use m for the sake of consistency. In the model

proposed by Nicola et al. (2019), we incorporate all the estimators unrelated to time windows,

including N , the (weighted) product of variances of x and y coordinates of vertices (V arX ∗

V arY ), the sum of distances to the nearest neighbor of each vertex (SumM inP ), the maximum

(maxDist) and sum (sumDist) of distances between customers, capacity (C), and the total

                             N
                                 
demand over capacity ratio   C
                                  .

     To make a fair comparison, we use the indicated predictors and linear regression models to

obtain the constants and coefficients for all six models. Our proposed regression models (5.3),

(5.4), and (5.5) are shown below:


                                       √
                    SDV RP (n) ≈ a + b1 nA + b2 B + b3 mean + b4 std,                        (5.3)


                                                 √
                              SDV RP (n) ≈ a + b1 nA + b2 B,                                 (5.4)


                                                77


--- PAGE BREAK ---

                             SDV RP (n) ≈ a + b1 mean + b2 std.                              (5.5)


      In the equations above, SDV RP (n) is the optimal solution value of the SDVRP solution.
     P             D(i)
B=      i∈V d(0, i) C , where C is the vehicle capacity and D(i) is the demand of customer i. B


can be viewed as a feature similar to the r̄n/Q feature in model (1), and 2B serves as a lower

bound on total length (Haimovich & Rinnooy Kan, 1985) and it describes the relationship be-
                                 √
tween customers and the depot.       nA is the square root of the number of vertices multiplied by

the area of the convex hull containing all the vertices and describes the inter-customer relation-

ships. mean and std are calculated based on the total lengths of the 1,000 random solutions. Note

that model (5.4) contains similar features as in Equation (5.1), and model (5.5) only contains the

features that come from the random solution procedure. The comparative analysis across these

models will disclose the important features for a robust regression model for the SDVRP.

                             Fitted Model                             Training MAPE   Test MAPE
 V RP (N ) = −6194.72 + 0.51N + 0.00V arX ∗ V arY + 6.19SumM inP
                                                                         257.33%       272.24%
           −9.51maxDist + 0.00sumDist − 2.06C −q    78.15 N
                                                          C
                                  −m
                                     √
     V RP (N ) = 3077.40 + 0.81 NN     N A − 1114.56 NA
                                                        − 8.81m          229.58%       198.77%
                                                  √
               V RP (N ) = 630.12 + 0.04r̄m√+ 0.03 N A                    33.26%        91.39%
              SDV RP (n) = 1004.51 + 0.51 nA + 1.97B                      56.91%        67.90%
                               √ + 1.00mean − 7.59std
             SDV RP (n) = 222.60                                          13.18%        14.37%
    SDV RP (n) = 18.19 + 0.79 nA + 0.19B + 0.60mean − 2.08std              2.91%         2.28%

                   Table 5.1: Different Models on the Training and Test Sets



      In Table 5.1, we randomly split 95 instances into a training set consisting 66 instances

(70%) and a test set containing 29 instances (30%). Subsequently, the training set was used to

estimate the coefficients and intercepts in the regression models. We report the mean absolute

percentage errors (MAPE) on the training and test set in the second and third columns, respec-

tively. We order the models by their MAPE values on the test set. Our first model, as represented


                                                 78


--- PAGE BREAK ---

by Equation (5.3), which includes both topological features, mean, and std achieves the mini-

mum MAPEs of within 3%. This indicates that to approximate route length totals in complex

routing problems like the SDVRP, using mean and std alone may not be enough, unlike our

results for the TSP and VRP (Kou, Golden, & Poikonen, 2024). We can also highlight that the

test MAPE value of the last model in Table 5.1 is substantially more accurate than 67.90% using
           √
only the       nA and B predictors and 14.37% using only the mean and std predictors, as well

as 91.39%, 198.77%, and 272.24% from the VRP regression models. This result indicates that

the VRP models from the literature may not be effectively extended to SDVRP instances. While

these VRP models have demonstrated high accuracies in the existing literature, they may not

be appropriate for VRP instances derived from the SDVRP, where multiple customers with unit

demand are sharing the same location.

                            Fitted Model                                      Adj.R2     MAPE
  V RP (N ) = 48.63 − 9.82N + 0.00V arX ∗ V arY + 5.50SumM inP
                                                                               0.998    762.68%
        −8.27maxDist + 0.00sumDist + 30.13C + q    1865.31 N
                                                           C
                               N −m
                                    √                A
  V RP (N ) = −3779.04 + 0.89 N      N A − 1533.74 N + 263.28m                 0.992    318.99%
                                                 √
              V RP (N ) = 161.63 + 0.04r̄m√+ 0.03 N A                          1.000    56.90%
              SDV RP (n) = 915.63 + 0.51 nA + 1.97B                            1.000    54.64%
                               √ + 1.00mean − 8.02std
             SDV RP (n) = 323.37                                               1.000    19.59%
  SDV RP (n) = −71.98 + 0.67 nA + 1.05B + 0.42mean − 5.18std                   1.000     3.07%

                    Table 5.2: Different Models on the Full SDVRP Instance Set



      Given our relatively limited instance set, we also utilize all 95 instances to estimate the

coefficients and intercepts. The resulting fitted models, along with their adjusted r-squared values

and MAPE values, are presented in Table 5.2. The order of different models are given by their

MAPE values on the entire instance set. From the last two columns, the high adjusted r-squared

values across all models confirm the strong linear relationship between predictors and the optimal


                                                79


--- PAGE BREAK ---

SDVRP solution value, aligning with the existing literature. Our model including mean, std, and

topological features in the sixth row consistently outperforms the rest without the randomness

introduced by the training-test split.

      Another observation we would like to highlight is the sign of coefficients in the models.

We have made similar observations in previous work: It should be intuitive that the optimal total

length is positively correlated with the mean of random total lengths and negatively correlated

with the std. This suggests that a higher mean indicates longer total length within the SDVRP

instances, while a higher std increases the likelihood of finding less costly or shorter solutions.



5.5    An Application

      In the previous chapter, we introduced a linear regression model for the VRP (Kou, Golden,

& Bertazzi, 2024b). This model incorporates the mean, standard deviation, and minimum of

random solution values as predictors. Within this section, we provide one application of the VRP

estimates based on our regression model: specifically, how they can help determine the stopping

time for the modified Clarke & Wright heuristic to obtain improved solutions. This is meant to

serve as a proof of concept, rather than a detailed algorithm enhancement. In this section, we

demonstrate the application to the VRP for simplicity, and note that it can be easily extended to

the SDVRP. In Kou, Golden, and Bertazzi (2024b), the regression model for the VRP instances

from the CVRPLIB is given by



                   V RP (n) = 0.531mean − 3.570std + 0.439min + 14.234,                        (5.6)




                                                 80


--- PAGE BREAK ---

where mean, std, and min denote the mean, standard deviation, and minimum of 1000 modified

Clarke & Wright solutions.

      We introduce an estimation enhanced, modified Clarke & Wright algorithm (EEMC&W),

detailed in Algorithm 2. The solution to the VRP instance is found by solving a set covering

problem using all feasible routes generated from the modified Clarke & Wright algorithm, as

mentioned in steps 4 and 5. In our implementation, we address the set covering problem using the

PuLP Package (Mitchell, OSullivan, & Dunning, 2011) incorporating the embedded CBC mixed-

integer linear programming solver (Forrest & Lougee-Heimer, 2005). We apply the modified

Clarke & Wright algorithm in which we consider the top 5 savings with equal probability. It is

important to note that the solution to the set covering problem may contain a single customer in

multiple routes. Nevertheless, since the Euclidean distance metric is used by instances from the

CVRPLIB, a shorter VRP tour can easily be derived from the set covering solution, ensuring each

customer is visited exactly once. Moreover, as outlined in step 6, Algorithm 2 will terminate if

either generating new possible routes no longer improves the current best solution and we have

already generated 10,000 modified Clarke & Wright solutions, or we find a solution with total

tour length less than or equal to the estimated value E ∗ . However, if the gap between the best

solution value and E ∗ exceeds 2% after generating 10,000 modified Clarke & Wright solutions,

additional modified Clarke & Wright solutions will be generated, as detailed in step 7.

      To understand the stopping condition of EEMC&W better, we present two examples in

Figure 5.3. For instance E-n76-k8 on the left, EEMC&W steadily improves solutions until gen-

erating fewer than 10,000 modified C&W solutions. Subsequently, it reaches a point where the

best solution remains unimproved for a while. However, given that the best solution is still more

than 2% away from the estimate (indicated by the dashed line), EEMC&W continues until ob-

                                               81


--- PAGE BREAK ---

Algorithm 2 Estimate Enhanced Modified Clarke & Wright (EEMC&W)

  1. For a given VRP instance, generate 1,000 modified C&W solutions.

  2. Compute the mean, standard deviation, and minimum from the 1,000 solution values. Sub-
     stitute the values of the predictors into equation (6), and obtain estimate E ∗ .

  3. Assign the number of modified C&W solutions generated to be N =1,000.

  4. Create a list of possible routes from the routes generated from the 1,000 modified C&W
     solutions.

  5. Assign previous best tour = None, previous best value = infinity, current best tour = VRP
     solution derived from solving the set covering problem over the possible routes, current
     best value = tour length of the current best tour.

  6. While the current best value is greater than E ∗ and N is smaller than 10,000 repeat the
     following steps.

        • Generate another 1,000 modified C&W solutions to enhance the list of possible routes
          and increase N by 1,000.
        • Update previous best value = current best value, and previous best tour = current best
          tour. Also recompute current best tour = the derived VRP solution from solving the
          set covering problem over the possible routes, and update current best value = tour
          length of the current best tour.
                                                                current best value−E ∗
  7. While current best value < previous best value and                   E∗
                                                                                         > 2%, repeat the
     following steps.

        • If N >20,000 and previous best value = current best value, then terminate.
        • Generate another 1,000 modified C&W solutions to enhance the list of possible routes
          and increase N by 1,000.
        • Update previous best value = current best value, and previous best tour = current best
          tour. Also recompute current best tour = the derived VRP solution from solving the
          set covering problem over the possible routes, and update current best value = tour
          length of the current best tour.

  8. Report the current best tour as the solution to the VRP.




                                              82


--- PAGE BREAK ---

                         Figure 5.3: EEMC&W on Two VRP Instances


taining 20,000 modified C&W solutions. Note that the EEMC&W stops at N =20,000 due to

the lack of improvement from N =19,000 to 20,000. Otherwise, N could be greater than 20,000

if continued improvement of the best solution were observed. In contrast, consider B-n45-k6

on the right of Figure 5.3, the EEMC&W obtains a better solution value than the estimate after

generating 6,000 modified C&W solutions, leading to its termination. Now, let us examine the

plot on the right in Figure 5.3 from a different perspective. Suppose the stopping rule without

an estimated solution is to stop at N = 10,000. With an estimated solution, we might stop at

N =6,000, because our best solution value is below the estimated solution value at N =6,000.

This might save a considerable amount of computing time.

      By measuring the gap between the current best solution and the estimate, the EEMC&W

provides an opportunity to discover better solutions even when the curve of the best value flattens

out. In Figure 5.4, we show instance P-n50-k8 as an example. Note that the best solution value is

not improved when N is between 5,000 and 14,000. However, notable improvements in the best

solution value are observed at N =15,000 and 19,000.

      In Table 5.3, we present numerical results of our proposed algorithm EEMC&W. For 93


                                                83


--- PAGE BREAK ---

                              Figure 5.4: EEMC&W on P-n50-k8


instances from the CVRPLIB with size 16 to 200, we compute the heuristic solutions using Al-

gorithm 2. We report the average computing time and the optimality gap. For comparison, we

also examine results obtained without utilizing the estimates. The modified Clarke & Wright

algorithm without estimate (MC&W10000 ) behaves similar to Algorithm 2, but MC&W10000 ter-

minates when we can no longer improve the best solution value after the number of modified

C&W solutions reaches 10,000. We observe that, compared to MC&W10000 , EEMC&W reduces

the average optimality gap without doubling the computational time.

             Algorithm      Average Computing Time (s)       Average Optimality Gap
            EEMC&W                    99.14                          1.14%
            MC&W10000                 57.24                          1.26%

                   Table 5.3: Application of Estimates as a Stopping Criteria




5.6   Conclusion

      The use of regression to estimate optimal total length in routing problems is a well-explored

area in the literature, with such estimates serving as benchmarks for evaluating heuristic al-


                                                84


--- PAGE BREAK ---

gorithms aiding companies in operational planning and in designing solution algorithms. This

chapter contributes to the field by developing regression models for the SDVRP. We introduce a

model that integrates topological features, mean, and standard deviation of random solution val-

ues, achieving an impressive accuracy with an error margin of approximately 3%. Our proposed

modified Clarke & Wright algorithm with split delivery (MCWSD) facilitates the quick genera-

tion of multiple random SDVRP solutions. We also provide an application section on using the

estimates to help the modified Clarke & Wright algorithm to find near-optimal solutions for the

VRP. Future research could focus on enhancing the MCWSD for better mean and standard devi-

ation predictors, generalizing the application to the SDVRP, or on extending similar regression

models to tackle even more complex routing problems.




                                              85


--- PAGE BREAK ---

Chapter 6:      The Application of Machine Learning Models to Optimal TSP Tour

                Length Estimation




6.1    Introduction

      As we have discussed in this dissertation, developing estimation models is an appealing ap-

proach to the Traveling Salesman Problem (TSP) for several reasons. Estimated tour lengths can

serve as valuable benchmarks for evaluating the performance of heuristic algorithms on new TSP

instances. Furthermore, these estimates are particularly useful in practical applications where

detailed routes are not essential. For instance, during the strategic planning phase for logistics

providers, decisions such as depot placement or service region size can be made independently

of specific daily routes, and the estimation results can also be useful in determining fleet size. In

addition, reliable TSP estimates can enhance the efficiency of algorithms tackling more complex

routing problems. For example, in vehicle routing problems (VRP), a common type of heuristic

is the cluster-first, route-second (CFRS) method, where effective and accurate approximation of

TSP tour lengths of a subset of vertices can facilitate better clustering of vertices.

      Sobhanan, Park, Park, and Kwon (2024) leverage a pretrained graph neural network (GNN)

to predict the objective function values of vehicle routing problems (VRPs), and these predicted

values are integrated into a genetic algorithm to efficiently solve hierarchical VRPs, such as



                                                 86


--- PAGE BREAK ---

the Multi-Depot VRP and the Capacitated Location Routing Problem. This work highlights

an innovative application of cost estimation models: enhancing the performance of solvers for

complex combinatorial optimization problems by evaluating subproblems quickly via estimation,

rather than solving them to optimality.

      Several approaches exist for approximating the total distance in transportation problems,

as we review in Section 6.2. As reviewed in previous chapters, early methods relied heavily on

analytically derived formulas (e.g., Beardwood et al. (1959)), sometimes supplemented by em-

pirically estimated parameters (e.g., Christofides and Eilon (1969)). However, these approaches

are typically limited to cases where vertex distributions are well-understood (e.g., uniformly dis-

tributed vertices), making them unsuitable for more complex instances. More recently, empirical

models, such as those proposed by Cavdar and Sokol (2015) and Nicola et al. (2019), have em-

ployed linear regression to build predictive models for optimal TSP tour lengths.

      Despite advancements in machine learning, few studies have applied machine learning

techniques to estimate optimal TSP tour lengths. Existing models either fail to achieve high

accuracy or struggle to generalize across different instances. Addressing these limitations, our

work proposes machine learning models that not only exhibit strong predictive performance but

also generalize well in a variety of problem settings. Specifically, we develop models using

linear regression (LR), random forests (RF), and neural networks (NNs) with carefully engineered

features. For the NN architecture, we leverage the Kolmogorov-Arnold Networks (KANs) (Liu

et al., 2024), a recent approach that is able to provide compact and interpretable models tailored

to the TSP.

      Beyond the aforementioned contributions, we conduct an analysis focused on the differ-

ences between machine learning models. Our study reveals that the Random Forest model pro-

                                                87


--- PAGE BREAK ---

vides high accuracy, but struggles with extrapolation. In contrast, KAN neural networks out-

perform the rest at generalizing to instances from unseen distributions within the experimental

dataset. These insights suggest a direction for future research to further investigate these differ-

ences and enhance model performance in logistics planning and other applications.

      The subsequent sections of this chapter delve into the details of our approach. Section 6.2

reviews related work on optimal TSP tour length approximation models. Section 6.3 presents the

experimental design, including the instances, features, and models. In Section 6.4, we discuss

our experimental results, compare our model’s performance with state-of-the-art approaches, and

analyze the differences among our models. Finally, Section 6.5 summarizes our findings and

suggests future research directions.



6.2    Literature Review

      In addition to the classic models we have reviewed in previous chapters, in recent years,

deep learning (DL) models have been developed to construct TSP tours (Vinyals, Fortunato, &

Jaitly, 2015; Kool, Van Hoof, & Welling, 2018). Despite this, relatively few studies have applied

machine learning techniques to predict the optimal tour length. To the best of our knowledge,

Kwon et al. (1995) were among the first to use neural networks for this purpose. Their model

achieves accurate estimates for instances with 10 to 80 vertices using variables like N , A1 , and

the ratio of the length to the height of the service region. More recently, Akkerman and Mes

(2022) applied several machine learning models, including linear regression, elastic net regres-

sion, random forest (RF), lightGBM (a gradient boosting method commonly applied in produc-

tion, built upon decision trees), and neural networks (NN), to estimate the optimal TSP and VRP



                                                88


--- PAGE BREAK ---

tour lengths. Their best performing model is the lightGBM and NN model using more than 30

features, which achieves testing MAPEs of around 5%.

       Varol, Özener, and Albey (2024) advanced this line of research, achieving high accuracy

in optimal tour length estimation with neural networks. For instance sizes of 20, 50, and 100

vertices, their models achieve MAPE values of 1.68%, 1.40%, and 1.08%, respectively. Despite

the high accuracy achieved by their model, its generalizability remains a concern. In particular,

they use 200, 000 instances of a particular size to train the NN model for such instances, and

the dimensionality of features depends on the instance size. Consequently, a different model is

required for each instance size. For example, we cannot predict the tour length of an instance

with 99 vertices using their trained models, even though 99 is close to 100. To predict the optimal

TSP tour length of an instance with 99 vertices, a separate model needs to be trained for it.

       Our contribution builds upon these works by exploring machine learning approaches for

estimating optimal TSP tour lengths. Our model achieves accuracy comparable to state-of-the-

art methods while also offering greater generalizability by using features that apply across a wide

range of instance sizes and vertex distributions. Detailed experiments are presented in Section

6.3.



6.3    Experiments

       In this section, we introduce the approximation model we propose for the TSP, along with

the computational experiments we conducted. We propose various features, and we provide the

details of feature engineering and selection in Section 6.3.1. Subsequently, in Section 6.3.2, we

describe the models implemented: (i) linear regression (LR), (ii) random forests (RF),(iii) Multi-



                                                89


--- PAGE BREAK ---

Layer Perceptron (MLP), and (iv) Kolmogorov-Arnold Networks (KANs), a specialized type of

neural network. We explain the generation of instances and data, as well as the hyperparameter

tuning process, in Section 6.3.3.



6.3.1     Feature Engineering


6.3.1.1      Random Solution Value-Related Features

        In earlier studies (Kou et al., 2022; Kou, Golden, & Poikonen, 2024), we identify the

mean (meanRT ) and standard deviation (stdRT ) of random tour lengths as effective predictors

for the optimal tour length in the TSP and VRP. However, we observed that the linear regression

model trained for the TSP did not achieve accuracy comparable to the model trained for the VRP

(Kou, Golden, & Poikonen, 2024). Notably, for the VRP, we used random heuristic solution

values instead of purely random feasible solution values. We hypothesize that this discrepancy in

performance is due to the higher quality of random heuristic solutions compared to completely

random solutions.

        To obtain higher-quality features, we use a simple heuristic to generate different feasible

solutions efficiently. We choose the 2-opt algorithm (Croes, 1958), a widely used local search

heuristic for improving TSP solutions. It works by iteratively removing two edges from the

current tour and reconnecting the tour in a way that reduces the total tour length. Instead of trying

to obtain better solutions, we start from a random tour, apply 2-opt until no more improvement

can be made, and record the 2-opt solution value. By starting from different random tours, the

2-opt algorithm helps us quickly generate better feasible solutions.

        Since we need a large number of random heuristic solution values to compute the meanRT


                                                 90


--- PAGE BREAK ---

and stdRT features, even generating one 2-opt solution can be costly as the instance size increases.

Therefore, we reduce the instance size first and then run the 2-opt algorithm on the smaller

instances. The reduction procedure follows the method described in Vakhutinsky and Golden

(1995). The idea is to partition the smallest rectangle containing all vertices into m × m smaller

disjoint rectangles (cells) of the same scale, where m is a user-defined parameter, and then use

the center of gravity in each non-empty cell as a new vertex in the reduced-size instance.

      It is easy to see that the maximum number of vertices in the reduced-size instances is

m2 . In fact, according to Vakhutinsky and Golden (1995), when there are n uniformly i.i.d.

vertices in the original instance, the expected number of vertices in the reduced-size instance is

m2 (1 − (1 − m12 )n ). For example, when there are 100 uniformly i.i.d. vertices in the original

instance and we use a 10 × 10 grid, we expect to have 63.4 vertices in the reduced-size instances.

      The algorithm to generate meanRT and stdRT features is detailed in Algorithm 3. The time

complexity of reducing the instance is O(n), and then the 2-opt algorithm given k vertices in the

smaller instance is O(k 2 ). The empirical computing time and comparative analysis with other

models in the literature are discussed in Section 6.4.



6.3.1.2    Lower and Upper Bounds

      In Kou, Golden, and Bertazzi (2024a), the addition of a spatial feature, which serves as a

lower bound on the optimal solution value, improved the predictive accuracy of the regression

model for the split delivery vehicle routing problems (SDVRP). Similarly, Varol et al. (2024)

include a partial solution feature in their models, which serves as a lower bound for the TSP.

Although computing such features is time-consuming, the improvement in predictive accuracy is



                                                91


--- PAGE BREAK ---

Algorithm 3 Generating meanRT and stdRT using 2-opt on reduced-size instances
Require: Set of vertices V , size of grid m, iterations for 2-opt t
 1: Step 1: Partition the Smallest Rectangle
 2: Find the smallest rectangle that contains all vertices in V
 3: Partition the rectangle into m × m smaller disjoint cells
 4: Step 2: Compute Centers for Non-Empty Cells
 5: for each cell in the m × m grid do
 6:     Find the vertices that lie within the cell
 7:     if the cell contains vertices then
 8:          Compute the center of the vertices in the cell
 9:          Add the center as a new vertex in the reduced-size instance Vreduced
10:     end if
11: end for
12: Step 3: Apply 2-opt Algorithm to the Reduced-size Instance
13: repeat
14:     Initialize the tour by random shuffling the vertices
15:     Perform a 2-opt swap that improves the tour length
16:     Record the tour lengths
17: until t 2-opt solutions are obtained
18: Step 4: Return the Predictor Values
19: Return the mean of the 2-opt solution values meanRT and the standard deviation of the 2-opt
    solution values stdRT




                                              92


--- PAGE BREAK ---

significant.

      We likewise incorporate upper and lower bounds as predictors to improve the accuracy of

our optimal tour length estimation models. Instead of using tight but computationally expensive

bounds, we choose bounds that are fast to compute.

      For the lower bounds, we generate 10 least-cost-1-trees, and the lower bound lbound is

determined by the longest one. A least-cost-1-tree can be obtained by finding the minimum

spanning tree (MST) for the remaining vertices and using two shortest edges to connect the

starting vertex with the MST. Generating least-cost-1-trees is efficient, and it is possible that a

least-cost-1-tree might be the optimal TSP tour. The time complexity of a least-cost-1-tree in a

complete graph is O(n2 log n) using Kruskal’s algorithm.

      For the upper bound, we run a single iteration of the 2-opt algorithm on the original in-

stance. As discussed earlier, the time complexity is O(n2 ), and the heuristic solution value ubound

is greater than or equal to the optimal TSP tour length L∗ .



6.3.1.3        Topological Features

      The literature review motivates us to include topological features in our models to further

improve predictive accuracy without over-complicating the models.

      We start by considering features mentioned in Akkerman and Mes (2022). In this chapter,

the authors summarize features from the previous literature and list 36 features for the TSP in-
               √
cluding N ,        N A, and their designed radius-related features and partition-related features. For

the latter, similar to what we have done to reduce the size of instances, the authors partition the

enclosing rectangle into 10 × 10 and 15 × 15 cells and consider spatial features like the average



                                                    93


--- PAGE BREAK ---

distances between vertices for vertices, within the same cell.

        In our feature selection, we want to include the random solution value-related features and

lower and upper bounds introduced in Sections 6.3.1.1 and 6.3.1.2. Thus, we uniformly select

the best k topological predictors using the Pearson correlation coefficient, then add the meanRT

and stdRT to train a linear regression model. We start with k = 1 and gradually increase until

the accuracy of the linear model is satisfactory. In the end, the two added spatial features are the
√
    N A and the perimeter of the convex hull p.



6.3.2      Models

        Using the carefully engineered features, we train four models: linear regression, random

forests, Kolmogorov-Arnold Networks (neural networks), and the classic feedforward Multi-

Layer Perceptron (MLP) neural network.

        The linear regression model is simple to train and interpret, and it has been widely used in
                                                                   √
previous research due to the linear relationship between L∗ and        N A. The LR model is also a

good starting point to compare with more complex models.

        Random forests are ensemble methods that build multiple decision trees during training and

aggregate their predictions. Each tree is trained on a random subset of data and features, making

the model robust to overfitting and capable of capturing complex relationships. We choose RF

because it can model complex, non-linear interactions between features and the target variable.

        Kolmogorov-Arnold Networks are based on the Kolmogorov-Arnold superposition theo-

rem, which states that any multivariate continuous function can be represented as a linear combi-

nation of univariate functions. KAN models use this principle to decompose the regression prob-



                                                  94


--- PAGE BREAK ---

lem and approximate complex functions with a series of univariate functions. Compared to the

more commonly used MLP, KAN has activation functions on the edges. KAN can also achieve

similar performance with fewer parameters, improving generalization. The detailed KAN applied

in our experiments will be explained in Section 6.4.2. We also include the results using MLP for

comparison purposes.



6.3.3      Computational Experiments

        To train our LR, RF, MLP, and KAN models, we follow the instance generation methodol-

ogy proposed by Varol et al. (2024), which utilizes arbitrary clusters. In this setup, the vertices are

not uniformly distributed; instead, they fall within a rectangular service region and are grouped

into elliptical clusters. We generate 10, 000 instances of size 20, 50, and 100, respectively. These

30, 000 instances are randomly divided into a training set of 24, 000 instances (80%) and a test set

of 6, 000 instances (20%). For comparison purposes, Nicola et al. (2019)’s LR model is trained

on the same set of instances, while we use the reported results of Varol et al. (2024)’s model

(trained on 100,000 instances of each size) as presented in their paper.

        Initially, the features described in Section 6.3.1 are computed for all instances. After per-

forming feature selection as detailed in Section 6.3.1.3, six features remain: meanRT , stdRT ,
                    √
lbound , ubound ,    N A, p. We train the LR, RF, KAN, and MLP models using these features and

the optimal TSP tour lengths obtained from Concorde. The LR, RF, and MLP models were im-

plemented using the scikit-learn package (Pedregosa et al., 2011), while the KAN model utilized

the authors’ open-source package (Liu et al., 2024).

        For hyperparameter tuning, we employ a grid search for both RF and NN models. Specif-



                                                  95


--- PAGE BREAK ---

ically, the set of 24,000 instances is further divided into a training set of 19, 200 instances and

a validation set of 4800 instances. We evaluate a set of potential hyperparameters and select the

ones that minimize the mean squared error on the validation set.

        For the RF model, we experimented with random forests using 10, 50, and 100 decision

trees, ultimately selecting 100 decision trees. For the MLP model, we explored configurations

with one hidden layer with 4, 8, 16, 32, or 64 hidden neurons; ReLU or logistic activation func-

tions; constant or adaptive learning rates; and regularization strengths of 0.0001, 0.001, 0.01, or

0.1. The final MLP model has 64 hidden neurons, ReLU activation, an adaptive learning rate,

and a regularization strength of 0.0001 among the 5 × 2 × 2 × 4 = 200 combinations. Note

that we did not conduct an exhaustive search, but explored a reasonable range of hyperparameter

combinations. Future research could focus on identifying the optimal neural network architec-

ture and fine-tuning hyperparameters to further improve performance. Hyperparameter tuning

was not performed for the KAN model, as Liu et al. (2024) did not provide tuning details in their

implementation, nor was it applied to the LR model. The final model weights are determined

using the training-validation set of 24,000 instances.



6.4     Results and Discussions


6.4.1     Choosing m for Instance Size Reduction

        To determine the appropriate value of m for instance size reduction, we focus on the per-

formance of the LR model. Figure 6.1 shows the mean absolute percentage error (MAPE) of

linear regression models trained using the predictors described in Section 6.3.1 on instances of

size 100. The only difference is the m × m grid we use to reduce instance size, and the resulting


                                                96


--- PAGE BREAK ---

                     Figure 6.1: MAPE Values of Linear Regression Models


2-opt solution values, which subsequently affect the meanRT and stdRT predictors. The hori-

zontal axis represents the number of effective vertices, i.e., the average number of vertices in the

reduced-size instances. The red dot corresponds to the original instance, without any reduction.

We observe that the LR model achieves MAPE values within 2% for the original instance; how-

ever, the computation time for these predictors becomes a concern as the 2-opt algorithm slows

down with larger instances.

      It is important to note that the 2-opt solution values for the reduced-size instances are

generally smaller than those in the original instances due to the reduced number of vertices.

Nonetheless, when comparing two distinct instances, the relationship between their meanRT

predictors is approximately preserved, according to Figure 6.2. Instances with larger optimal

TSP tour lengths tend to have correspondingly larger 2-opt solution values in the reduced-size

instance. This consistency allows the meanRT predictor obtained from the reduced-size instances

to remain effective. Although the relationship between the stdRT predictors is less stable under

instance size reduction, we retain it to complete the model.

      Figure 6.3 illustrates the relationship between the grid coarseness and the number of effec-



                                                97


--- PAGE BREAK ---

                            Figure 6.2: Comparing meanRT Predictors


tive vertices, as well as the average computation time required to generate 100 2-opt solutions for

original instances of size 100. We observe that, while the computational time for 2-opt solutions

increases linearly with grid coarseness, the number of effective vertices grows at a slower rate.

Combining Figure 6.1 with Figure 6.3, we aim to balance computational cost and model accu-

racy. Consequently, we select a 10 × 10 grid for instance reduction. For our largest instances

of size 100, this configuration results in approximately 50 effective vertices in the reduced-size

instances, reducing the 2-opt computational time to less than half of that required for the original

instances.



6.4.2     Model Performance

        Using the methodology described in Section 6.3.1, we train five models: a linear regres-

sion (LR) model, a random forest (RF) model with 100 decision trees, two Kolmogorov-Arnold

Network (KAN) models with one hidden layer of size 16 and 4, and a Multi-Layer Perceptron

(MLP) neural network with one hidden layer of size 64. We also experimented with KAN models

                                                98


--- PAGE BREAK ---

                       Figure 6.3: MAPE Values of Linear Regression Models


featuring 32 and 64 hidden neurons, but observed minimal improvement in model accuracy along

with increased training time. Therefore, we report only the KAN models with 4 and 16 hidden

neurons in this section.

      Table 6.1 presents the MAPE values on the test set described in Section 6.3.3 for these

models, along with comparisons to the models proposed by Varol et al. (2024) and Nicola et al.

(2019), both of which are prominent in the literature. The Varol et al. (2024) model represents the

state-of-the-art neural network (NN) approach with high estimation accuracy, while the Nicola et

al. (2019) model is the most recent linear regression model, other than our models.

                   Model         Testing MAPE        Feature Generation Time (CPU)a
                     LR              2.44%                        168
                     RF              1.72%                        168
                    MLP              1.51%                        168
                  KAN(16)            1.55%                        168
                   KAN(4)            2.19%                        168
                  Varol NNb          1.39%                        177
                  Nicola LR          6.20%                        11.6
a
  The Run Time is compared for generating features for 1,000 instances of size 100 using one CPU; b The result
comes from the average of three models


            Table 6.1: Result of Different Models on Instances of Size 20, 50, and 100



                                                     99


--- PAGE BREAK ---

      It is important to note that the Varol et al. (2024) model is restricted to instances of a fixed

size. To facilitate a meaningful comparison, we calculate the average of MAPE values across

different sizes in Table 6.1 and provide a breakdown for different instance sizes in Table 6.2. For

the Nicola et al. (2019) model and our models, we train them on 8, 000 instances of each fixed

size and present the test results on the remaining 2, 000 instances in Table 6.2. Since the Nicola

et al. (2019) model was not trained on our dataset, we retrain their linear regression model using

the same predictors described in their paper and include the results here for comparison.
                           Instance Size     Model        Testing MAPE
                                               LR             2.10%
                                               RF             1.79%
                                              MLP             1.42%
                                 20
                                            KAN(16)           1.56%
                                             KAN(4)           1.69%
                                            Varol NN          1.68%
                                            Nicola LR         5.38%
                                               LR             1.94%
                                               RF             1.87%
                                              MLP             1.52%
                                 50
                                            KAN(16)           1.54%
                                             KAN(4)           1.64%
                                            Varol NN          1.40%
                                            Nicola LR         5.43%
                                               LR             1.93%
                                               RF             1.88%
                                              MLP             1.52%
                                100
                                            KAN(16)           1.39%
                                             KAN(4)           1.45%
                                            Varol NN          1.08%
                                            Nicola LR         6.34%

                 Table 6.2: Different Models Trained on Instances of Fixed Size



      As shown in Tables 6.1 and 6.2, our LR model outperforms the Nicola et al. (2019) model,

underscoring the importance of effective feature engineering and reinforcing the strong relation-

ship between the distribution of feasible solutions and the optimal solution value for the TSP.

                                                100


--- PAGE BREAK ---

While our more complex machine learning models do not surpass the performance of the Varol

et al. (2024) model, they achieve comparable predictive accuracy with the added advantage of

being applicable to instances of varying sizes. In contrast to the MLP model, the KAN model

achieves comparable MAPE values with a significantly smaller number of hidden neurons, which

indicates that using the KAN architecture, we can obtain a highly accurate yet compact model

with a small amount of training time. For fixed-size instances (Table 6.2), we observe that fine-

tuning these models for specific instance sizes can further improve accuracy.

      Although the KAN model did not outperform the traditional MLP architecture in terms

of prediction accuracy, it offers several advantages. The KAN architecture provides more inter-

pretability and insight into the “black-box” nature of neural networks. To better understand the

KAN model, we train a version starting with only four hidden neurons using the training set for

Table 6.1 and prune the model by removing less important edges. We then fine-tune the pruned

model further to enhance estimation accuracy. Figure 6.4 visualizes this pruned model based on

a specified edge-importance threshold. Black dots represent neurons, while boxes with curves

indicate activation functions applied to the edges. After pruning, three hidden neurons remain.

Additionally, activation functions connected to the leftmost hidden neurons were deemed less

important and are not shown in the figure. In KAN, these activation functions are splines with

user-defined degrees. For our experiments, we set the degree to 3 and observe that most activation

functions appear linear (except the one connecting stdRT which is a reciprocal function), which

corroborates the linear relationship between predictors and the optimal TSP tour lengths, L∗ .

      This simplified model, though being less accurate than the KAN models in Tables 6.1

and 6.2, enables us to extract a closed-form relationship between the predictors and the optimal

TSP tour length using symbolic regression. To uncover the analytical relationship between the

                                               101


--- PAGE BREAK ---

    Figure 6.4: Visualization of the Pruned KAN Model Starting with Four Hidden Neurons


features and the predicted value L∗ , we follow the methodology proposed by Liu et al. (2024).

The derived relationship from KAN(4) is approximated as follows:


                                                                 √
                L∗ = 262.13lbound + 229.13meanRT + 93.68p + 26.37 N A

                                    634.22
                             + (−0.06stdRT −1)
                                              2 + 287.66ubound + 463.79                      (6.1)



which is unsurprisingly almost linear, given that the activation functions are mostly linear. One

thing to note is that the symbolic regressions are highly sensitive and one may not get the same

result each time based on the random initialization of the network. To confirm this almost linear

relationship between the predictors and the predicted value, we obtain multiple analytical ex-

pressions following the steps described in Liu et al. (2024). Although we get different analytical

expressions, the derived functions are similar to the one of LR in equation (6.2) in terms of the

sign and magnitude of the coefficients.



                                               102


--- PAGE BREAK ---

      The visualization allows the user to determine the function type of each activation func-

tion. Thus, if researchers have prior knowledge of the relationship between the features and the

predicted value, they can fix the function type at certain edges. In our example, we fix the ac-
                                        √
tivation function associated with the    N A feature to be linear, based on the Beardwood et al.

(1959) result. The simplified KAN(4) model shown in Figure 6.4 has a testing MAPE of 2.4%,

demonstrating that even with fewer parameters, KAN models can offer both interpretability and

competitive performance.

      It is also insightful to analytically compare the relationship between the predictors and L∗

as derived from both the KAN(4) model and the LR model. Equation (6.2) is the LR model we

obtained. Notably, the coefficients of the linear terms in the LR model not only share the same

signs as in the KAN(4) model but also exhibit similar magnitudes. For the stdRT feature, which

is non-linear in equation (6.1), L∗ is negatively correlated with it in both cases.


                                                                   √
                  L∗ = 313.04lbound + 215.42meanRT + 99.15p + 59.04 N A

                     − 87.92stdRT + 223.54ubound + 1101.08                                   (6.2)



      To assess the near-linear behavior of the KAN(16) model’s predictions, we constructed a

linear regression model using the input features and the predicted values of the KAN(16) model,

                                                                                    linear
from Table 6.1. Let yKAN represent the predicted values from the KAN(16) model and yKAN

denote the predictions from this linear regression model. The resulting r-squared value of 0.999

indicates an almost perfect linear relationship, providing strong evidence that the black-box KAN

model demonstrates near-linear behavior with respect to the input features.




                                                103


--- PAGE BREAK ---

6.4.3     The Ability to Generalize

        To highlight the generalizability of our models and explore the differences between various

machine learning approaches, we conduct a generalization study in this section. Using the same

models trained on instances of size 20, 50, and 100 in the previous section, we estimate the

optimal TSP tour lengths for instances of size 80, 120, and 150, which are created in the same

manner. We generate 1, 000 instances for each new size and report the testing MAPE values in

Table 6.3.

        Additionally, we evaluate the models on a set of challenging Euclidean TSP instances, de-

noted as Tn,m , introduced by Hougardy and Zhong (2021). These “tetrahedron instances” are

parameterized by n and m, representing the number of vertices along the side and inside the pro-

jection of a tetrahedron, respectively. The Tn,m instances with N ∈ {52, 55, 58, ..., 199} exhibit

distinct vertex distributions and are computationally intensive to solve exactly using Concorde.

Two of the Tn,m instances are visualized in Figure 6.5. For instance, solving the N = 199 in-

stance (on the right of Figure 6.5) optimally with Concorde requires over 400, 000 seconds on

a 2.2GHz Intel Xeon E5-2699, according to Hougardy and Zhong (2021). This makes Tn,m an

ideal test set to assess the robustness of the models discussed in this article in scenarios signif-

icantly different from the training set. The testing MAPE values for these 50 instances are also

presented in Table 6.3.

        To further evaluate the model’s generalization ability, we introduced an additional set of

test instances with an unseen vertex distribution. Specifically, we generated 1, 000 instances of

size 100, using a 2-dimensional triangular distribution within a 500 × 500 square. The MAPE

values for these 1, 000 instances are summarized in Table 6.3. The two-dimensional triangular


                                                104


--- PAGE BREAK ---

                                 Figure 6.5: Two Tn,m Instances




     Figure 6.6: Two 2-Dimensional Triangular Distributed Instances and the Distribution


distribution is similar to the bivariate normal distribution in which vertices are clustered in the

center; an example instance is shown on the left side of Figure 6.6. To further illustrate this

distribution, we plotted a size-10,000 instance in the middle of Figure 6.6. Like the bivariate

normal distribution, the triangular distribution is dense near the center and becomes sparser as it

extends outwards. However, the triangular distribution does not have smooth tails (see the right

of Figure 6.6). The testing MAPE values for these 1, 000 instances are also included in Table 6.3.

      From the results in Table 6.3, we observe that the estimation accuracy of all models on the

out-of-sample test sets is generally lower than the in-sample results shown in Table 6.1, which is

expected since the test instances differ from those in the training set. For the LR and RF models,


                                               105


--- PAGE BREAK ---

the testing MAPE values increase as we move from instances of size 80 to 120 and 150. This

trend indicates that, as the test instances deviate more from the training set, model performance

degrades. The smallest decrease in accuracy occurs for instances of size 80, which is closest in

size to the training instances (sizes 50 and 100), whereas the largest instances (size 150) show the

highest MAPE values due to their dissimilarity to the training set.

      The Tn,m instances, with their unique structure and vertex distributions, pose a significant

challenge for all models except the KAN model. The RF model, which is the only non-parametric

model, struggles to generalize, showing a substantial increase in MAPE. Unlike the parametric

models, non-parametric models do not assume a fixed number of parameters. Instead, the com-

plexity of the model grows with the data size, allowing them to capture more complex relation-

ships. We experimented with RF models with different hyperparameters, and this deterioration

in MAPE values is consistent. Our experiments indicate that the relationship between the pre-

dictors and the optimal TSP tour length, which is well captured by the RF model for the training

set, is not transferable to the Tn,m instances. In fact, RF models are highly effective at capturing

complex patterns in the data but are inherently limited to the feature representations seen dur-

ing training. They rely on decision tree splits, which can struggle with extrapolation beyond the

training distribution. NNs are more flexible in learning abstract representations. While the MLP

and KAN models also experience reduced accuracy on these instances, the magnitude of degra-

dation is considerably smaller compared to the RF model, suggesting a better capacity to adapt to

unseen instances. We hypothesize that the KAN model captures complex functional relationships

between features and predictions with fewer weights. Notably, when the MLP model’s hidden

layers are increased to 3, the test MAPE on the Tn,m instances drops below 2%; this aligns with

our conjecture.

                                                106


--- PAGE BREAK ---

      We observe a similar pattern for the triangular distributed instances, where the RF model

struggles to generalize to unseen instances. However, these triangular instances are more closely

aligned with the training set compared to the Tn,m instances. The MLP model demonstrates its

ability to capture relationships between predictors and the predicted optimal tour length, achiev-

ing higher predictive accuracy than the LR model. Among all models in our experiment, the

KAN model outperforms the others, exhibiting the best generalization across different instances.

      In summary, Table 6.3 demonstrates that different models have distinct strengths. Among

the models, both the MLP and KAN demonstrate robust performance on instances generated

similarly to the training set, highlighting the ability of neural networks to capture complex re-

lationships between predictors and the optimal TSP tour length. However, when tested on the

structurally different Tn,m instances, the simpler LR model, despite having fewer parameters,

proves to be more robust than the MLP. This robustness likely stems from its reliance on a linear

relationship, making it less sensitive to the nuanced differences within the instance set.

      In contrast, when the vertex distribution closely resembles the training distribution (e.g.,

80, 120, 150, and triangular instances), the MLP model surpasses the LR model in performance.

Meanwhile, the KAN model better adapts to the unseen instances in the training set, which is

quite impressive. Liu et al. (2024) show that the KAN model generalizes more effectively than

the MLP when approximating mathematical functions. Our experiments provide an evidence

of KAN’s robustness in more complex setups. Although the predictions are not as accurate on

instances similar to the training set, the degree of deterioration is smaller than that of the other

models.

      Complex models like NNs and RFs excel when the test instances share similar charac-

teristics with the training set, but are more sensitive to deviations. Conversely, the simpler LR

                                                107


--- PAGE BREAK ---

model, while less accurate on the training-like instances, shows greater resilience when faced

with substantially different test instances, such as the Tn,m instances.

                            Instance Set     Model      Testing MAPE
                                              LR            6.78%
                                              RF            2.80%
                                 80          MLP            1.82%
                                            KAN(16)         1.86%
                                            KAN(4)          2.25%
                                              LR            8.05%
                                              RF            3.84%
                                120          MLP            1.85%
                                            KAN(16)         2.01%
                                            KAN(4)          2.45%
                                              LR            8.80%
                                              RF            4.96%
                                150          MLP            2.14%
                                            KAN(16)         2.19%
                                            KAN(4)          2.65%
                                              LR            6.29%
                                              RF           99.56%
                                Tn,m         MLP            6.67%
                                            KAN(16)         1.21%
                                            KAN(4)          1.61%
                                              LR           14.42%
                                              RF           20.49%
                             Triangular      MLP           12.82%
                                            KAN(16)         2.05%
                                            KAN(4)          6.15%

                           Table 6.3: Out-of-Sample Evaluation Results




6.4.4     Model Performance on the Entire Set

        The differences among various models raise an important question: If we include Tn,m in-

stances and triangular instances in the training set, can the performance of models that previously

struggled to generalize improve? To study this, we combine the new instances described in Table



                                                108


--- PAGE BREAK ---

6.3 with the 30, 000 instances of sizes 20, 50, and 100 described in Table 6.2, creating a dataset

of 34, 050 (30, 000 + 3, 000 + 50 + 1, 000) instances. This dataset is then divided into training

and test sets. To analyze how quickly the models learn the relationship between features and the

optimal TSP tour length, we vary the training set size, using randomly selected subsets of 20%

and 80% of the entire dataset.

      To avoid redundancy, Tables 6.4 and 6.5 present the MAPE values of various models for

the test set and for subsets of test instances when training on 20% and 80% of the dataset, respec-

tively.

    Model          All      20, 50, 100      80         120       150     Tn,m      Triangular
    LR           15.93%      16.63%       14.69%      10.52%    10.12%    0.38%      12.10%
    RF           2.01%        1.78%        1.72%       1.68%     1.80%    7.03%       1.11%
    MLP          7.95%        8.09%        5.80%       4.73%     4.41%    0.30%      12.75%
    KAN(16)      6.31%        6.48%        5.34%       6.26%     7.25%    0.48%       2.04%

                  Table 6.4: Mixed Distribution MAPE Results (20% Training)


     Model          All     20, 50, 100      80         120      150      Tn,m     Triangular
     LR          14.35%      15.59%        13.60%     10.10%    9.81%     0.28%     10.79%
     RF           1.72%       0.86%        0.96%       0.87%    0.84%     1.22%      0.58%
     MLP          7.75%       7.78%        5.73%       4.54%    4.05%     0.19%     14.40%
     KAN(16)      4.83%       4.87%        3.76%       4.73%    5.37%     0.37%      3.96%

                  Table 6.5: Mixed Distribution MAPE Results (80% Training)


      From Tables 6.4 and 6.5, we observe that as the amount of training data increases, all

models improve their ability to learn the relationship between features and the optimal TSP tour

length. Machine learning models, in particular, capture more complex relationships and consis-

tently outperform the LR model. Interestingly, although the RF model previously struggled with

extrapolation (Section 6.4.3), including instances from all distributions in the training set enables

it to learn quickly, achieving an average MAPE of 2.01% with just 20% of the dataset (Table 6.4).


                                                109


--- PAGE BREAK ---

Moreover, the RF model maintains low MAPE values across all instance distributions included

in the training set.

      In contrast, the MLP and KAN models show difficulty in learning relationships when com-

bining different distributions (by comparing MAPE values in Tables 6.4 and 6.5 with those in

Table 6.1), potentially due to the increased complexity that may no longer be expressible using

simple parametric models. However, both neural network models balance errors across differ-

ent vertex-distributions, with KAN(16) slightly outperforming MLP. Another noteworthy obser-

vation is that both the LR and MLP models (using ReLU activation) achieve somewhat lower

MAPE values on larger instances. This trend, shown in the 80, 120, and 150 columns of Ta-

bles 6.4 and 6.5, aligns with the asymptotic convergence of linear relationships demonstrated by

Beardwood et al. (1959) and Kou et al. (2022).

      The results presented in Sections 6.4.3 and 6.4.4 highlight how to select among the models

based on the real-world environment. If the decision maker is confident that future instances will

follow a similar distribution to the training data, the RF model would be a good choice due to its

ease of training and strong performance on similarly distributed data. However, if the decision

maker is uncertain about the distribution of new instances, the KAN model may be more suitable,

as it has demonstrated robust generalization capabilities, at least on the test instances analyzed in

our study.



6.5    Conclusion

      This chapter introduces a novel machine learning approach (KAN) to estimating the op-

timal tour length for the TSP, demonstrating improvements over traditional empirical methods.



                                                110


--- PAGE BREAK ---

Our study shows that machine learning models—especially random forests (RF) and neural net-

works (NNs)—deliver superior predictive accuracy across a diverse range of TSP instances. The

adoption of Kolmogorov-Arnold Networks (KANs) further enhances our methodology by offer-

ing compact and interpretable models that effectively capture some of the complexities of our

estimation problem.

      Our models not only achieve competitive accuracy compared to existing models in the lit-

erature but also exhibit strong generalizability across different problem sizes and distributions.

We also present an efficient feature engineering framework that balances computational cost and

model performance, enabling scalable and practical application to large-scale TSP instances. Ad-

ditionally, our generalization study and analysis of model performance offer insights into select-

ing appropriate models for practical applications, depending on the objective and the distribution

of instances.

      Overall, this work advances TSP research by leveraging machine learning to bridge the gap

between traditional analytical approaches and the growing demand for accurate, scalable solu-

tions in combinatorial optimization. Future research can build on these findings by integrating our

models into advanced routing heuristics, potentially leading to more efficient and cost-effective

solutions in logistics, transportation, and beyond. For example, it would be natural to apply our

models to solve hierarchical vehicle routing problems. Moreover, theoretical investigations into

the differences among machine learning models in the context of vehicle routing could provide

new insights.




                                               111


--- PAGE BREAK ---

Chapter 7:     Accurately Estimating Optimal SDVRP Solution Values using Re-

               gression Models




7.1   Introduction

      The Split Delivery Vehicle Routing Problem (SDVRP), introduced by Dror and Trudeau

(1989), extends the classic Vehicle Routing Problem by allowing customer demands to be met

by multiple deliveries. This generalization can lead to significant cost savings and operational

efficiencies in logistics by optimizing vehicle capacity utilization and reducing the total travel

distance of all vehicles (Archetti & Speranza, 2008). In particular, the potential savings from

allowing split deliveries can reach up to 50% in the worst case (Archetti et al., 2006). However,

this increased flexibility also dramatically expands the solution space, causing exact methods to

become computationally prohibitive for medium- and large-sized instances. Consequently, there

is substantial value in accurate and computationally efficient methods to estimate the optimal

solution value (or total distance) for the SDVRP.

      Reliable estimation models for the optimal SDVRP total distance offer several critical ben-

efits. Firstly, they provide robust benchmarks against which heuristic solutions can be evaluated,

guiding algorithmic improvements. Secondly, logistics companies can use these estimates in

strategic decision-making, assessing potential cost savings from adopting split deliveries versus



                                               112


--- PAGE BREAK ---

traditional vehicle routing. Furthermore, accurate estimates are practical in complex multi-depot

or multi-day problems, allowing pre-assignment of customers to depots or days, thus, simplifying

subsequent route optimization (Sobhanan et al., 2024). From a theoretical perspective, optimal

solution estimation builds upon classical studies dating back to the seminal work by Beardwood

et al. (1959) on estimating optimal tour lengths in the Traveling Salesman Problem (TSP).

      In earlier work on the SDVRP (Kou, Golden, & Poikonen, 2024), we introduced a regression-

based approach to estimate the optimal total distance for SDVRP instances. Our model relied on

key predictors such as the mean and standard deviation of random (simple) heuristic solution val-
                                      √
ues, alongside a topological feature ( nA) and a lower bound derived from depot-to-customer

distances. This linear regression framework not only delivered accurate estimates but also pro-

vided an interpretable analytical expression, making it valuable for benchmarking heuristic al-

gorithms and supporting strategic decision-making. In related work on the TSP (Kou, Golden,

& Bertazzi, 2025), we explored various machine learning approaches to assess both predictive

accuracy and model generalizability, while also gaining insights into the functional relationships

between engineered features and the optimal solution value. This chapter is inspired by both our

earlier work and the recent work of others (Varol et al., 2024), which demonstrated the effec-

tiveness of sophisticated neural network architectures for estimating optimal TSP tour lengths.

In contrast, we address a more challenging combinatorial optimization problem, achieve com-

parable predictive accuracy with simpler models, and show that our estimates exhibit greater

robustness.

      Building on these results, the present study leverages a newly generated dataset of SDVRP

instances. These instances were randomly generated and solved to near-optimality using a state-

of-the-art heuristic SplitILS (Silva et al., 2015). At the time of its publication, SplitILS achieved

                                                113


--- PAGE BREAK ---

an average improvement of 1.15% over previously best-known solutions, and recent compara-

tive studies continue to recognize it as a leading benchmark heuristic in the SDVRP literature

(Gamst, Lusby, & Ropke, 2024). The availability of near-optimal solutions across a wide variety

of instances enables a more rigorous evaluation of multiple approaches, including linear regres-

sion, random forests, and neural network architectures such as multilayer perceptrons (MLP) and

Kolmogorov-Arnold networks (KAN) (Liu et al., 2024).

      Our contributions are twofold. First, we rigorously quantify the incremental improvements

in predictive accuracy, measured by mean absolute percentage error (MAPE), achieved by adopt-

ing more complex machine learning models compared to simpler linear regression models. Sec-

ond, we critically examine whether these marginal accuracy improvements justify the associated

loss of interpretability and increased computational complexity. Our experiments demonstrate

that linear regression remains a compelling choice due to its simplicity, interpretability, and com-

petitive performance, especially important in practical applications where ease of interpretation

and implementation is critical. The reason for this, we think, is the fact that although linear re-

gression is comparatively simple, identifying the right predictor variables is not an easy task, and

the predictor variables that we use are very effective.

      In Section 7.2, we detail our methodology for instance generation and optimal solution

benchmarking. This is followed by the development and evaluation of various regression models

in Section 7.3. Next, in Section 7.4, we explore the trade-offs between model complexity and

interpretability by assessing model performance on both the primary set of randomly generated

instances and an extended set of instances of different sizes. As a result of this study, we aim

to provide robust models for accurate SDVRP total distance estimation and offer insights into

the optimal balance between accuracy and interpretability in predictive modeling for complex

                                                114


--- PAGE BREAK ---

routing problems like the SDVRP.



7.2   Instance Generation and Benchmarking

      To evaluate our regression models comprehensively, we generated an extensive dataset of

SDVRP instances. Using instance generation techniques inspired by Queiroga et al. (2021), we

generated 2,160 Euclidean problem instances, with 60, 80, and 100 customers, covering a wide

range of parameters, including depot location, customer spatial distributions, and heterogeneous

demand levels. For detailed instance generation procedures, readers may refer to Queiroga et al.

(2021). To ensure the generated instances are suited for the SDVRP, customer demands were set

relatively high. Specifically, we selected demand types with large values and either high or low

coefficients of variation, following the approach in Queiroga et al. (2021). Additional details are

provided in Appendix C.

      Each instance was solved using the state-of-the-art SplitILS heuristic (Silva et al., 2015),

which has demonstrated an optimality gap of around 0.2% in recent studies (Gamst et al., 2024).

We chose to generate our own instances rather than rely on the limited number of benchmark

instances commonly used in the SDVRP literature, as a few hundred instances may be insufficient

for training machine learning models, particularly neural networks. Inspired by Queiroga et al.

(2021), who developed a set of 10,000 VRP instances to support machine learning evaluations, we

generated over 2,000 SDVRP instances for the same purpose. These near-optimal solutions serve

as high-quality benchmarks, providing excellent targets for training and evaluating our regression

models. The SplitILS-derived total tour lengths form the target variable for prediction, allowing

us to assess estimation accuracy effectively.



                                                115


--- PAGE BREAK ---

      Given our focus on practical, real-world applicability and our dependence on SplitILS, our

models are designed to estimate the best achievable SDVRP solution value, rather than optimal

solution values. In fact, while having exact optimal solutions for all 2,160 SDVRP instances

would enable highly precise models for estimating the true optimal value, this remains infea-

sible. Exact algorithms are limited to relatively small instances: typically no more than 100

customers (Archetti & Speranza, 2008). Moreover, even for the DIMACS benchmark instances

used in many studies, only best-known (not proven optimal) solutions are available. Bertazzi and

Wang (2022) provide a comprehensive study comparing matheuristic approaches with worst-case

performance bounds on 36 benchmark instances. Their work highlights the scalability limitations

of exact solvers and demonstrates how near-optimal matheuristics can improve the best-known

solutions for the SDVRP. Therefore, we choose to adopt the efficient and scalable SplitILS as our

reference solver.

      This rich dataset of over 2,000 SDVRP instances facilitates the training of more complex

models, such as random forests and neural networks, while enabling a rigorous comparison with

the simpler linear regression model. To evaluate model generalization, we split the dataset into

distinct training (80%) and testing (20%) subsets, ensuring that performance metrics reflect both

in-sample fit and out-of-sample predictive accuracy.

      In addition to the previously generated instances of sizes 60, 80, and 100, we incorporated

144 new Euclidean instances of sizes 70 and 90 (288 in total) in Section 7.4.3. These were gen-

erated using the same parameter combinations described in Appendix A. Using models trained

on the original instance sizes (60, 80, and 100), we made direct predictions on the new instances

to evaluate the generalizability and robustness of the models.

      In summary, the combination of a large, diverse set of generated SDVRP instances and

                                               116


--- PAGE BREAK ---

near-optimal solutions from SplitILS forms the foundation of our experimental framework. This

robust dataset is essential for exploring the trade-offs between the incremental accuracy gains

offered by complex models and the inherent interpretability of the linear regression approach.

To further enhance our evaluation, we extend the experiments to new instances of different sizes,

providing a more comprehensive understanding of the regression models’ performance on various

problem instances.



7.3     Regression Models

        The goal of this work is to develop regression models that accurately estimate the opti-

mal total tour length for SDVRP instances. To achieve this, we explored multiple modeling

approaches, ranging from simple linear regression to more complex machine learning models.

This section details the feature selection, model development, and evaluation methodology.



7.3.1     Feature Selection

        In prior research (Kou, Golden, & Poikonen, 2024), we identified four key features that

have strong predictive power in the linear regression model through stepwise regression. To

incorporate potential non-linear relationships between features and the optimal SDVRP total dis-

tance, we trained a random forest model with an extensive set of features commonly used in

the VRP optimal tour length estimation literature. We then selected the top six features with

the highest importance scores (this is a metric that shows how much each feature contributes to

the model’s accuracy (Breiman, 2001)), in the random forest model for training the regression

models.



                                               117


--- PAGE BREAK ---

     The features we selected for consideration, based on either stepwise regression or the high-

est importance scores from the random forest model, can be broadly categorized into two groups:


   • Statistical features from random solutions: These features capture statistical properties of

      1,000 random feasible SDVRP solutions generated using the modified Clarke & Wright

      algorithm as detailed in Kou, Golden, and Poikonen (2024).


        – mean: The average total distance across the 1,000 generated solutions.

        – std: The standard deviation in total distance, which captures the variability among

            feasible solutions.

        – min: The shortest total distance observed among the 1,000 solutions.

        – max: The longest total distance observed among the 1,000 solutions.

        – Weibull Lower Bound (BW eibull ): An estimated lower bound obtained by fitting a

            Weibull distribution to the 1,000 generated random solution values, motivated by the

            distribution’s flexibility and finite lower bound, which makes it well-suited for ap-

            proximating the distribution of feasible solution values.


   • Instance attribute-related features: These features describe the spatial distribution of cus-

      tomers and demand characteristics relative to vehicle capacity constraints.

            √
        –       nA: The square root of the product of the number of customers (n) and the area (A)

            of the convex hull enclosing all customers, serving as a proxy for customer dispersion.
            Pn
                i=1 Di
                                                         Pn
        –        C
                         : The ratio of total demand (    i=1 Di ) to vehicle capacity (C), which repre-


            sents the lower bound on minimum number of required vehicles.



                                                   118


--- PAGE BREAK ---

                                   Di
                    P
           – B=2        i∈V d(0, i) C : A depot-to-customer distance-weighted demand metric, cal-


             culated as the sum of the products of two times depot-to-customer distances (d(0, i))

             and customer demands (Di ), normalized by vehicle capacity (C). This serves as a

             theoretical lower bound for the SDVRP tour length, as proposed by Haimovich and

             Rinnooy Kan (1985).


        Some of these features were selected based on their strong linear correlations with the op-

timal total tour length, while others were chosen based on their importance scores in the random

forest model.



7.3.2     Model Development

        We implemented and evaluated four regression approaches, resulting in 6 different models,

each offering a different balance of accuracy and interpretability.



7.3.2.1      Linear Regression

        As a baseline, we employed a linear regression model derived from the training set using

the selected four features identified through stepwise regression, following the approach in Kou,

Golden, and Poikonen (2024). This model provides a closed-form analytical expression, enabling

direct interpretation of how each feature contributes to the estimated tour length. The fitted linear

model, trained with raw features, takes the form:


                                      √
                      L = 55.03 + 0.01 nA + 0.27B + 0.71mean − 0.48std.                        (7.1)




                                                119


--- PAGE BREAK ---

L is the optimal SDVRP solution value that we try to estimate. This model achieved a mean

absolute percentage error (MAPE) of approximately 1.01% on the generated test instances of

size 60, 80, and 100, demonstrating strong predictive performance despite its simplicity.

      To facilitate comparison with the abstract formula derived from the Kolmogorov-Arnold

network (KAN) model (see Section 7.4.1), we standardized (so that each has a mean of 0 and a

standard deviation of 1) all features in the training set and re-estimated the model coefficients.

Standardizing the features does not affect the predictive accuracy of the linear regression model,

as it simply re-expresses the same linear relationship in a different coordinate system. The stan-

dardized linear regression model is expressed as:


                                 √
            L = 17052.74 + 103.15 nA + 2378.36B + 6386.43mean − 33.67std.                     (7.2)



This transformation allows for a fair comparison with the KAN model, where feature standard-

ization is implemented to facilitate model convergence.

      To ensure a fair comparison across different regression approaches, we trained an additional

linear regression model using the top six features identified based on feature importance scores

from a random forest model, as described in Section 7.3.2.2. It is important to note that these

selected features may not exhibit a linear relationship with the optimal solution value. A detailed

comparison of this model with others is presented in Section 7.4.2.



7.3.2.2    Random Forest Regression

      To capture potential non-linear relationships between predictors and tour lengths, we em-

ployed a random forest regression model. This ensemble method constructs multiple decision


                                               120


--- PAGE BREAK ---

trees on bootstrapped subsets of the data and averages their predictions to improve robustness

(Breiman, 2001). Hyperparameter tuning was conducted using Bayesian optimization, adjusting

parameters such as the number of trees, maximum depth, and minimum samples per leaf.

      The final random forest model, trained on the top six features identified from feature im-
                                                      Pn
                                                       i=1 Di
portance scores (min, max, mean, B, BW eibull , and     C
                                                                ) achieved an MAPE of 0.90% on the

test set. While this model demonstrates strong predictive accuracy, it lacks interpretability, as it

does not provide an explicit functional form.



7.3.2.3     Multilayer Perceptron (MLP) Neural Network

      To further explore non-linear modeling, we trained a feedforward neural network using a

multilayer perceptron (MLP) architecture. The MLP model consisted of a single hidden layer

with ReLU activation functions and was optimized using the L-BFGS algorithm with an initial

learning rate of 0.001. Bayesian optimization was applied to fine-tune hyperparameters, including

the number of neurons and the regularization rate.

      Using the same feature set (the six features with the highest importance scores) as in the

random forest model, the best-performing MLP model achieved an MAPE of 0.76%, reflecting

modest gains in accuracy compared to simpler models. While the MLP network performed well,

its black-box nature limits interpretability.



7.3.2.4     Kolmogorov-Arnold Network (KAN)

      To bridge the gap between traditional neural networks and interpretable analytical expres-

sions, we implemented Kolmogorov-Arnold network (KAN) models (Liu et al., 2024). Unlike



                                                121


--- PAGE BREAK ---

conventional neural networks, KAN models apply trainable activation functions to the edges

connecting neurons, allowing them to capture complex relationships while extracting an explicit

analytical form. We trained two KAN models using distinct feature sets: one based on step-

wise regression and the other selected from random forest feature importance scores. The first

model produced an analytical expression that closely aligned with equation (7.2) and achieved

an MAPE of around 0.92%, highlighting the predominantly linear relationship between features

and the optimal solution value as discussed in Section 7.4.1. The second model, incorporating

six features selected from the random forest model, provided deeper insights into potential non-

linear interactions with an MAPE of 0.79% on the test set. A detailed discussion of the second

model is presented in Section 7.4.2.



7.4    Results and Discussion

      This section presents the results of our evaluation of the regression models on both the

test set of generated SDVRP instances of sizes 60, 80, and 100, and the extension set of sizes

70 and 90. In Section 7.4.1, we compare a linear regression model and a KAN model trained

using the same set of four predictors as in Kou, Golden, and Poikonen (2024), validating the

linear relationship between these predictors and the optimal total distance. Section 7.4.2 includes

a comparison of four models—linear regression, random forest, MLP, and KAN—each trained

on the top six features identified by a random forest model. Finally, Section 7.4.3 evaluates the

generalizability of these four models on the extension set. We compare model performance based

on predictive accuracy and interpretability, providing insights into the trade-offs.




                                                122


--- PAGE BREAK ---

                   Figure 7.1: KAN Model Using Stepwise Regression Features



7.4.1     Model Comparison Using Step-Wise Regression Features

        To examine the relationship between selected features through stepwise regression in Kou,

Golden, and Poikonen (2024) and the optimal SDVRP total distance, we trained a KAN model

with two hidden neurons allowing for easier extraction of its analytical form. The resulting model

is illustrated in Fig. 7.1. The four predictor variables are at the bottom and optimal solution (L) is

at the top of Fig. 7.1. The KAN model initializes the activation functions on the edges as splines.

As one can observe, the trained activation functions look mainly piecewise linear.

        This model achieved an MAPE of 0.92% on the test set, demonstrating a slight improve-

ment over the linear regression model presented in equation (7.2) by incorporating non-linearities.

To obtain a closed-form expression, we prune less significant neurons if any exist. There are none

in this experiment and we extract an analytical expression in (7.3) that closely resembles equation


                                                 123


--- PAGE BREAK ---

                 Figure 7.2: Final KAN Model Using Stepwise Regression Features


(7.2):


                                    √
               L = 17037.90 + 111.19 nA + 2365.23B + 6353.15mean − 27.43std.                   (7.3)


         Following the default settings described in Liu et al. (2024), the functional forms become

fixed after extracting an analytical expression and are visually highlighted in red in the KAN

model plot. This pruned KAN model is depicted in Fig. 7.2 and yields an MAPE of 0.98%.

The extracted formula closely aligns with the linear regression model, indicating that the SDVRP

tour length estimation problem remains largely linear in nature with the selected set of features.

The minimal non-linearity captured by the KAN model provided only marginal improvements,

reinforcing the effectiveness of a simpler linear formulation. This suggests that, given the selected

features, additional complexity does not significantly enhance predictive accuracy with respect


                                                 124


--- PAGE BREAK ---

to our SDVRP dataset.



7.4.2     Model Performance on Generated SDVRP Instances

        To assess the predictive performance of each model and compare their performance on the

same set of features, we trained the regression models on the training subset (80%) of the 2,160

generated SDVRP instances and evaluated them on the remaining 20% test set using the six

features selected from the random forest importance scores, instead of features obtained from the

stepwise regression as detailed in Section 7.4.1. The set of features used in this experiment are
                                     Pn
                                      i=1 Di
mean, min, max, B, BW eibull , and     C
                                               . To allow for improved accuracy, we used a KAN model

with four hidden neurons in this experiment, as opposed to the simpler model in Section 7.4.1.

Table 7.1 summarizes the mean absolute percentage error (MAPE) for each model.

        Table 7.1: MAPE results across regression models on generated SDVRP instances.

                Model                      MAPE (Training Set)       MAPE (Test Set)
                Linear Regression               0.86%                   0.90%
                Random Forest                   0.37%                   0.90%
                MLP Neural Network              0.73%                   0.76%
                KAN Model                       0.77%                   0.79%


        The linear regression model achieved an MAPE of 0.90% on the test set, serving as a strong

baseline. Despite its simplicity, it performed competitively against more complex models. The

random forest model achieved the lowest MAPE (0.37%) on the training set but exhibited a no-

ticeable increase to 0.90% on the test set, indicating potential overfitting. While the MLP neural

network and KAN models achieved slightly lower test MAPEs of 0.76% and 0.79%, respectively,

their improvement over linear regression was small.

        Although the complex models provided slight gains in predictive accuracy, these improve-


                                                    125


--- PAGE BREAK ---

            Figure 7.3: KAN Model Using Selected Features by Feature Importance


ments came at the cost of interpretability. To better understand how the selected features con-

tribute to predicting optimal total distances, we look into the activation functions in the KAN

model, shown in Fig. 7.3.

      Unlike in previous experiments with stepwise regression features, the activation functions

in Fig. 7.3 exhibit more obvious non-linear behavior, suggesting that the selected features capture

more complex interactions. However, this added complexity does not translate into a substantial

reduction in MAPE, suggesting that the relationship between features and optimal solution value

remains largely linear.

      Furthermore, the increased complexity introduces additional computational costs. Notably,

computing the added feature BW eibull requires fitting a Weibull distribution to the set of random

feasible solutions, making feature generation less straightforward. These findings suggest that



                                               126


--- PAGE BREAK ---

while advanced machine learning models can marginally enhance accuracy, they do so at the

expense of transparency, interpretability, and ease of deployment.



7.4.3     Generalization to Size 70 and 90 Instances

        While the neural network models in Varol et al. (2024) achieve high accuracy in estimat-

ing optimal TSP tour lengths, it has been noted that their models cannot be directly applied to

instances with a different size from the training set. In Section 7.4.2, we provide models that are

more general and trained on instances of size 60, 80, and 100. To further assess the generalization

ability of the models, we tested them on 144 instances of size 70 and another 144 instances of

size 90. Rather than retraining the models, we applied them directly to these new instances to

predict the optimal solution values and compute the corresponding MAPE values. The results are

summarized in Table 7.2.

        Table 7.2: MAPE results across regression models on size 70 and size 90 instances

                   Model                   MAPE (Size 70)      MAPE (Size 90)
                   Linear Regression          0.76%               0.74%
                   Random Forest              1.00%               0.89%
                   MLP Neural Network         0.75%               0.68%
                   KAN Model                  0.89%               0.67%


        The linear regression model demonstrates strong robustness across both extension sets.

Notably, the MAPE values on instances of size 70 and 90 are even lower than the 0.90% test

MAPE reported in the previous section, highlighting the model’s generalizability. This reinforces

the effectiveness of simple linear models for SDVRP optimal solution estimation, particularly in

applications where interpretability and computational efficiency are important.

        The random forest model yielded the highest MAPE values in the generalizability study:


                                               127


--- PAGE BREAK ---

1.00% MAPE for size 70 and 0.89% MAPE for size 90 instances, suggesting its limited ability to

generalize to unseen data. Nevertheless, the relatively low error indicates that instances generated

under the same methodology retain structural similarities, despite differences in the number of

customers.

         The MLP neural network and KAN models exhibited comparable results, consistent with

findings in (Liu et al., 2024). The MLP outperforms the linear regression model on both instance

sets, reflecting its capacity to generalize and capture nonlinear relationships. Similarly, the KAN

model achieved the lowest MAPE on the size 90 instances. However, the improvement over linear

regression is marginal, suggesting that the simpler linear model remains a strong and practical

candidate for SDVRP optimal solution estimation.

         An additional observation is that all four models performed better on the size 90 instances

than on the size 70 instances. This pattern is consistent with trends observed in many TSP articles

(Beardwood et al., 1959; Varol et al., 2024; Kou et al., 2022), where theoretical results show

for the TSP that a constant multiple of a predictor asymptotically converges to the optimal tour

length, and this asymptotic property helps explain the improved regression performance on larger

instances. The observed results for the SDVRP suggest that a similar asymptotic relationship may

exist.



7.4.4      Discussion and Future Directions

         Our findings have several implications for SDVRP estimation and related logistics appli-

cations. For operational planning requiring transparency and rapid deployment, linear regression

remains a robust choice, offering reliable estimates with interpretable coefficients. This simplicity



                                                 128


--- PAGE BREAK ---

makes it particularly useful for decision-makers who prioritize clarity and ease of understanding.

          In scenarios where higher accuracy is essential, such as benchmarking advanced heuristics

or guiding algorithmic improvements, the KAN model best balances performance with inter-

pretability and the MLP model provides the highest estimation accuracy. The ability of MLP and

KAN to capture non-linear relationships enhances both predictive power and practical usability.

          Moreover, the dominance of solution-derived features, such as the mean, standard devia-

tion, and minimum of feasible solution values, underscores the utility of generating high-quality

random solutions during instance preprocessing. Additionally, incorporating spatial features such
     √                                                                        Pn
                                                                                i=1 Di
as       nA, B, and the lower bound on the number of required vehicles,          C
                                                                                         , is essential when

trying to obtain highly accurate estimation models for complex routing problems like the SDVRP.
                             Pn
                               i=1 Di
Although both B and             C
                                        serve as lower bounds, on the total tour length and the minimum

number of required vehicles, respectively, they capture complementary signals of the problem.
                 Pn
                   i=1 Di
Specifically,       C
                            reflects the lower bound on minimum number of routes necessary to meet

total demand, while B incorporates spatial information to approximate the minimal travel dis-

tance. Including both in regression models enhances predictive performance by combining both

insights.

          In the generalization study, we found that while MLP and KAN models may be preferable

when accuracy is the highest priority, the linear regression model remains a strong contender due

to its simplicity, ease of training, and competitive predictive performance.




                                                       129


--- PAGE BREAK ---

7.5   Conclusion

      In this study, we developed and analyzed regression models for estimating the optimal

total distance of SDVRP instances, leveraging a large dataset of generated instances solved to

near-optimality using the SplitILS heuristic. We compared multiple modeling approaches, rang-

ing from multiple linear regression to more complex machine learning models, including ran-

dom forests, multilayer perceptron (MLP) neural networks, and Kolmogorov-Arnold networks

(KAN).

      Our findings show that, while advanced machine learning models can achieve slight im-

provements in predictive accuracy, these gains come at the cost of increased complexity and

reduced interpretability. The linear regression model, despite its simplicity, remained highly

competitive, achieving a test set MAPE of 0.90% on the generated dataset with 60, 80, and 100

customers. When evaluated on two extension sets with 70 and 90 customers, the linear regression

model maintains strong performance, achieving a test MAPE of 0.76% and 0.74% respectively.

In one study, the pruned KAN model produced an analytical expression very similar to the linear

regression formula, demonstrating the almost linear nature of SDVRP optimal solution estimation

when using carefully selected features.

      In our generalization study on SDVRP instances of sizes 70 and 90, all four models, linear

regression, random forest, MLP, and KAN, achieve impressively low MAPE values, with better

performance on size 90 instances. While the MLP and KAN models showed slightly higher ac-

curacy, the linear regression model remained highly competitive, offering strong generalizability

and the advantages of simplicity and interpretability.

      These results emphasize the trade-off between accuracy and interpretability in SDVRP esti-

                                               130


--- PAGE BREAK ---

mation models and the importance of thinking about which predictor variables to include. While

machine learning models provide some improvements, they introduce additional computational

costs and reduce interpretability. For scenarios where efficiency and interpretability are crucial,

linear regression remains a compelling choice. Another important takeaway (emphasized by

Table 7.2) is that if estimation models can work this well for the SDVRP, a very difficult com-

binatorial optimization problem, then they might be successfully applied to a wide variety of

challenging problems in transportation and logistics. Future research could explore hybrid ap-

proaches that incorporate non-linear transformations while maintaining interpretability, further

enhancing SDVRP optimal solution estimation for real-world logistics applications.




                                               131


--- PAGE BREAK ---

Chapter 8:      Conclusion and Future Directions



      This thesis extends prior research by demonstrating that random feasible solution statistics

serve as effective predictors for optimal solution values across various combinatorial optimization

problems, including the Traveling Salesman Problem (TSP), Vehicle Routing Problem (VRP),

and Split Delivery Vehicle Routing Problem (SDVRP). We confirm that the mean, standard de-

viation, and minimum of random feasible or random heuristic solutions provide accurate tour

length estimations, offering an alternative stream of predictors than more traditional topological

features.

      Additionally, we systematically evaluated the trade-offs between interpretability and pre-

dictive power by comparing linear regression, random forests (RF), multilayer perceptron (MLP)

neural networks, and Kolmogorov-Arnold Networks (KAN) on TSP and SDVRP instances. The

insights gained not only enhance tour length estimation accuracy but also provide valuable guid-

ance on model selection for different problem settings.

      Chapter 2 focused on tour length estimation for the TSP, extending the work of Basel and

Willemain (2001) by validating the power-law relationship between the standard deviation of ran-

dom feasible tours and the optimal tour length. We established a theoretical connection between
                         √
this predictor and the    N A predictor, demonstrating its effectiveness across both Euclidean and

non-Euclidean instances using TSPLIB datasets, randomly generated instances, and real-world



                                                132


--- PAGE BREAK ---

data.

        Chapter 3 expanded this framework to other combinatorial optimization problems, includ-

ing the VRP, Minimum Spanning Tree (MST), and Maximum Weight Matching (MWM). We

improved the estimation model by incorporating a second predictor, the mean of random feasi-

ble solution values, which further enhanced predictive accuracy. For the VRP, heuristic-based

solutions from a modified Clarke & Wright algorithm were used to obtain statistical predictors.

Additionally, we refined the regression model by introducing the minimum of heuristic solution

values, achieving further improvements in accuracy.

        Chapter 4 explored an alternative feature extraction approach for non-Euclidean TSP in-

stances using Sammon mappings (Sammon, 1969). This method was applied to real-world road

network data from São Paulo, Brazil, and compared with traditional circuity-based models. The

results showed that Sammon mappings effectively capture topological structure, maintaining high

estimation accuracy while offering visualization benefits.

        Chapter 5 extends our estimation framework to the Split Delivery Vehicle Routing Prob-

lem (SDVRP), a more complex variant of the VRP. We introduce a modified Clarke & Wright

algorithm (MCWSD) designed to incorporate split deliveries, describe the feature engineering

process, and evaluate the accuracy and interpretability of the resulting models. Using established

DIMACS benchmark instances, we demonstrate that the model achieves high predictive accuracy,

with an error margin of approximately 3%. Additionally, this chapter explores how regression-

based estimators can enhance heuristic solvers for the VRP, illustrating a practical integration of

estimation and optimization.

        Chapter 6 introduced machine learning-based models for TSP tour length estimation, eval-

uating the performance of linear regression, random forests (RF), multilayer perceptrons (MLP),

                                               133


--- PAGE BREAK ---

and Kolmogorov-Arnold Networks (KAN). We demonstrated that RF models perform well on

seen distributions but struggle with extrapolation, whereas KAN models generalize effectively

to unseen distributions, making them a promising tool for scalable and interpretable tour length

estimation.

      Chapter 7 focused on estimating SDVRP tour lengths using a newly generated dataset of

2,160 SDVRP instances, solved to near-optimality via the SplitILS heuristic (Silva et al., 2015).

Regression models, including linear regression, RF, MLP, and KAN, were compared, revealing

that machine learning models provided slight accuracy gains, but linear regression remained a

strong candidate due to its efficiency and interpretability.

      This work motivates several directions for future research:

    • Extending Estimation Models to Other Combinatorial Optimization Problems: While this

      thesis focused on the TSP, VRP, and SDVRP, the underlying methodologies could be ap-

      plied to other routing problems or combinatorial optimization problems. Adapting statisti-

      cal predictors to these settings could provide valuable optimal cost estimates.


    • Incorporating Estimation Models into Heuristic and Metaheuristic Algorithms: The ac-

      curate estimation models developed in this work could be leveraged within heuristic al-

      gorithms for multi-day or multi-depot VRPs, using Genetic Algorithms (Sobhanan et al.,

      2024), to guide search direction and improve computational efficiency.


    • Hybrid Machine Learning and Optimization Approaches: Future studies could explore

      combining ML-based estimations with integer programming solvers (like we did in Section

      5.5), potentially leading to new hybrid frameworks that enhance solution quality.


    • Real-World Applications in Logistics and Transportation: Applying these estimation tech-

                                                134


--- PAGE BREAK ---

      niques to real-world routing problems in dynamic environments or on road map instances

      could help decision-makers optimize operational planning more effectively.


      In summary, this thesis demonstrates that random feasible solution statistics provide a pow-

erful, generalizable, and computationally efficient approach to tour length estimation across mul-

tiple combinatorial optimization problems. By bridging statistical methods, heuristic approaches,

and machine learning techniques, this work offers practical tools for logistics optimization and

lays the foundation for future advancements in routing and combinatorial optimization.




                                               135


--- PAGE BREAK ---

Appendix A:         Properties of Standard Deviation of Random Tour Lengths



      By extending Basel and Willemain (2001) work to more instances in the TSPLIB, we find
                      √
that both stdRT and       N A can be high-quality estimators of the optimal tour length for Euclidean
                                                                                      √
instances. Thus, we would like to explain the connection between stdRT and                N A when we

have the Euclidean metric. Let the set of vertices be V = {v1 , v2 , ..., vN }, and let the set of edges

connecting vertices vi and vj be {ei,j }i̸=j∈{1,2,...,N } . Let ci,j denote the corresponding network

distance of edge ei,j . Assume that there exists a finite value r such that 1r ≤ ci,j ≤ r, for all

i ̸= j ∈ {1, 2, ..., N }. Let us formally define the random neighbor procedure (RNP) as follows.

Algorithm 4 Random Neighbor Procedure (RNP)
  vcurrent ← v1
  S ← {v1 }
  E←∅
  while V \ S ̸= ∅ do
      Select vnext uniform randomly from V \ S
      S ← S ∪ {vnext }
      E ← E ∪ {ecurrent,next }
      vcurrent ← vnext
  end while
  E ← E ∪ {ecurrent,1 }



      The essence of the RNP is that it constructs a random closed tour of the vertex set by

choosing an unvisited neighbor at random at each iteration. Then, the output set of edges E

contains all edges that have been visited in this random tour.

      In previous research, several authors have studied the distribution of the optimal TSP tour

                                                  136


--- PAGE BREAK ---

lengths when the vertices are i.i.d. uniformly distributed. Beardwood et al. (1959) argued that

one can expect that the distribution of L∗ approaches a normal distribution as N → ∞ based on

the Central Limit Theorem. Vinel and Silva (2018) shown that when vertices are i.i.d. uniformly

distributed in a unit square or a unit circle, even for small N = 4, L∗ exhibits a normal distribu-

tion. However, to our knowledge, no authors have studied the distribution of random tours when

N → ∞.


Theorem A.0.1. Given the assumptions above, as N → ∞, the distribution of tour lengths

generated by RNP converges to a normal distribution.


Proof. Suppose µ is the average length of a tour in a graph. Let L1 , ..., LN be random variables

representing the lengths of edges selected via the RNP, where Li is the length of the selected edge

emanating from vi . Let E1 , E2 , ..., EN each be variables, taking an integer value from 1 to N, that

indicate edges selected under the RNP. In particular, if Ei = j, then the edge from vi to vj has

been selected as part of the random tour. Thus, if Ei = j, then Li = ci,j . To make this clear,

for each i = 1, ..., N , Ei reveals the next vertex after vi in the random tour, and Li specifies the
                                                               PN
length of the edge from vi to the next vertex. Suppose L =        i=1 Li , then L is a random variable


representing the full length of a random tour. Now let Xi = E(L − µ|E1 , ..., Ei ); that is to say, let

Xi represent the expected deviation of random tour length L from the average µ given that the first

i edges have already been selected. Note that E(Xi+1 − Xi |X1 , ..., Xi ) = 0 and |Xi+1 − Xi | ≤ r

by assumption. Thus, the sequence Xi is a martingale, and by applying the martingale central

limit theorem, we find that XN converges in distribution to the normal distribution. Therefore,

L = µ + XN also converges in distribution to the normal distribution.


      To demonstrate Theorem 1, for 20 vertices, we generated a complete distance matrix with

                                                 137


--- PAGE BREAK ---

               Figure A.1: Distribution of Random Tour Lengths for 20 Vertices




    Figure A.2: Variance in Random Tour Lengths versus Sum of Variance in Edge Lengths


all values independently uniform distributed between 0 and 1. In Figure A.1, we have the distri-

bution of 1, 000, 000 random tour lengths generated by the RNP, and we observe it to be approx-

imately normal.
                                                                                       √
      Now, we want to identify the connection between the stdRT predictor and the          N A pre-

dictor for Euclidean instances on a compact set. Consider the simplest example, where v1 , ..., vN

are uniform randomly distributed in a two-dimensional compact set. We considered instances

with N = 4, 5, ..., 200 vertices and took 100,000 random tours for each instance. Figure A.2

shows the relationship between variance in random tour lengths, V ar(L), and sum of variances
                                            PN
in edge lengths connected to each vertex,     i=1 V ar(Li ), for the unit square and unit disk. We

                                                        PN
can observe a linear relationship between V ar(L) and     i=1 V ar(Li ).




                                               138


--- PAGE BREAK ---

      Note that, for uniformly i.i.d. vertices, Li are identically distributed, and we can denote

V ar(Li ) := σ 2 for i = 1, ..., N . Based on the empirical results above, as we take k random tour

lengths for large k and calculate the standard deviation, it is reasonable to argue that

                                     v
                                     u N
                        p            uX              q         √
               stdRT   = V ar(L) ≈ C t   V ar(L ) = C N σ 2 = C N σ
                                         1                  i      1      i     1               (A.1)
                                                 i=1


                                      √              √        √
where C1 is some constant, and the     N term matches N in the N A predictor.
                                             √
      Now, we want to associate σ with           A, the square root area of the convex hull containing

all vertices. Kendall, Last, and Molchanov (2009) stated that, for any convex set with volume A0

in two-dimensional Euclidean space, A asymptotically converges to


                                      2rA0 log(N )      log log N
                               A0 −                + O(           )                             (A.2)
                                        3    N              N


where r is the number of vertices on the convex hull. First, consider a square with side length a.

For v1 , ..., vN uniformly i.i.d. in the square with side length a, Har-Peled (2011) proved that



                                        E(r) = O(2 log N ).                                     (A.3)


                                                       √               √
Combining Equation (A.2) and (A.3), we have             A converging to A0 = a. Next, we will

show that there exists a strong linear relationship between a and σ. According to Philip (2007),

for v1 , v2 i.i.d. uniformly distributed on the square with side length a, the squared distance d2




                                                    139


--- PAGE BREAK ---

between v1 , v2 , has the following p.d.f.

                       
                          √
                       −4 a3s + aπ3 + as4                                  0 < s ≤ a2
                       
                       
             g(s) =                                                                                (A.4)
                       
                                               √
                       − 22 + 42 arcsin √a + 43 s − a2 − π2 − s4
                       
                                                                            a2 < s ≤ 2a2 .
                           a     a            s      a             a    a



Philip (2007) also points out the following formula:


                                             Z 2a2
                                                     √
                               E(d) =                 sg(s) ds
                                              0
                                        a       √    1     √
                                      =   ln(1 + 2) + (2a + 2a2 ).                                 (A.5)
                                        3            15


        With


               Z 2a2
    2
E(d ) =              sg(s) ds
                0
               Z a2      √                   Z 2a2
                        4 s   π     s                   2     4         a     4√            π     s
         =          s(− 3 + 3 + 4 ) ds +           s(− 2 + 2 arcsin √ + 3 s − a2 − 2 − 4 ) ds
                0        a    a     a         a2       a     a           s a                a     a
                                                           Z 2a2 √                  Z 2a2
                  19      π        6 + 3π 2 7 2          4                        4                 a
         =     (− a2 + a2 ) + (−          a − a )+ 3             (s s − a2 ) ds + 2       (s arcsin √ ) ds
                  15       2           2        3       a a2                     a a2                s
                 89                   8               1               1
         =     − a2 − πa2 + (8a2 − ) + 8a2 arcsin         − 2a2 arcsin + 4
                 15                   3              2a               a
                                                           Z 2a2 √                  Z 2
                  19 2 π 2         6 + 3π 2 7 2          4                        4 2a              a
         =     (− a + a ) + (−            a − a )+ 3                      2
                                                                 (s s − a ) ds + 2        (s arcsin √ ) ds
                  15       2           2        3       a a2                     a a2                s
               31 2                     1              1 4
         =        a − πa2 + 8a2 arcsin    − 2a2 arcsin + ,                                           (A.6)
               15                      2a              a 3


we are able to see that the variance in d, σ 2 = E(d2 ) − E(d)2 , depends linearly on a2 . Because
                                                                                                     √
A converges to a2 as N → ∞, σ 2 has a linear relationship with A as N → ∞. Thus, σ and                   A

have an asymptotic linear relationship when v1 , ..., vN are i.i.d. uniformly distributed in a square.
                                                  √
This linear relationship between σ and             A also holds when we have vertices uniformly i.i.d. on


                                                         140


--- PAGE BREAK ---

a disk with radius R. According to Har-Peled (2011), for v1 , ..., vN uniformly i.i.d. in a disk


                                                           1
                                            E(r) = O(N 3 ).                                     (A.7)


                                                       √                                    √
Combining Equation (A.2) and (A.7) yields that             A converges asymptotically to     πR2 . In

addition, according to Lellouche and Souris (2020), if the vertices generation follows a Poisson

point distribution, which means locations of vertices only depend on the uniform distribution,

then



                               σ 2 = V ar(Li )         for i = 1, ..., N
                                                27 2 2
                                    = (1 − (       ) )R
                                               45π
                                    ≈ 0.180R2 .                                                 (A.8)


                                                                                        √
Thus, if we have vertices i.i.d. uniformly distributed on a disk with radius R, σ ≈         0.180R2 =
  √        √
C2 πR2 ≈ C2 A for large enough N and some constant C2 . To summarize, for i.i.d. Euclidean

instances on the unit square or unit disk with N sufficiently large, asymptotically, there is a proven
                                    √                    √                          √
linear relationship between σ and       A, and stdRT ≈ C1 N σ. Therefore, stdRT ≈ C3 N A, for
                                          √
some constant C3 . This suggests that      N A and stdRT may provide comparable estimates for

Euclidean i.i.d. instances with large N .




                                                 141


--- PAGE BREAK ---

Appendix B:       Mean and Standard Deviation for the MST and MWM



      For a given undirected graph on N vertices, the minimum spanning tree (MST) is a set of

N − 1 edges that connects all vertices in the graph such that the sum of the edge lengths is mini-

mized. The left side of Figure B.1 shows an example of the MST in two-dimensional Euclidean

space, where the red solid edges are in the MST and the black dashed edges are not. In our

experiments, we focus on complete graphs. According to Chazelle (2000), classic algorithms,

like Prim’s algorithm or Kruskal’s algorithm, run in O(m log N ) time, which is O(N 2 log N ) for

complete graphs. In addition, Chazelle (2000) and Pettie and Ramachandran (2002) both pro-

vided a best known upper bound of O(mα(m, N )) for their algorithms, where α is the functional

inverse of Ackermann’s function. Applying either classic or faster algorithms, we can solve MST

problems in polynomial time.

      A matching in a graph is a set of pairwise non-adjacent edges, and the nd maximum weight

matching (MWM) considers a matching in which the sum of edge lengths is maximized. The

right side of Figure 2 shows an example of the MWM in two-dimensional Euclidean space,

where the red solid edges are selected in the MWM. According to Duan and Pettie (2014), the

best known time complexity for the MWM problem is O(N 3 ) when a complete graph with non-

integer weights is given.




                                               142


--- PAGE BREAK ---

                              Figure B.1: Example of MST and MWM


      Our experiment is designed as follows. The weight of the MST, T , is computed using the

MST function from the Scipy package (Virtanen et al., 2020). For each instance, we generate

1,000 random spanning trees as feasible solutions to the MST and then compute the mean and

the standard deviation in their weights as meanST and stdST . For MWM problems, we use the

NetworkX package (Hagberg et al., 2008) to compute the maximum weight M , and 1,000 random

perfect matchings are considered as feasible solutions. We use meanM and stdM to denote the

mean and the standard deviation in the weights of perfect matchings. We use linear regression

to fit models (B.1) to (B.4) below, and we find the numerical values for the constant coefficients

a1 , a3 , b1 , b3 , and c3 for each MST instance set and constants a2 , a4 , b2 , b4 , and c4 for each MWM

instance set:

                                    ln(T ) = a1 × ln(stdST ) + b1                                   (B.1)


                                    ln(M ) = a2 × ln(stdM ) + b2 .                                  (B.2)


                                T = a3 × meanSP + b3 × stdSP + c3                                   (B.3)


                               M = a4 × meanM + b4 × stdM + c4 .                                    (B.4)


For the MST and MWM, we can solve for their optimal solutions easily with limited computing

                                                  143


--- PAGE BREAK ---

power. In addition, their instance structures are similar to those of the TSP and VRP, where we

have either an x-y coordinate system for the vertices or a given distance matrix. Hence, if we can

show that the power-law relationship between the standard deviation of random solution values

and the optimal solution value holds for these two problems, we will be more confident in answer-

ing yes to the following question: Does the standard deviation predictor work for optimization

problems other than the TSP?



B.1     Results for the MST

       To compare the experimental results of models in the form of Equations (B.1) and (B.3),

we summarize the experimental results and the values of fitted coefficients in Table B.1 and Table

B.2.

       Our uniform instances are 90 uniformly distributed instances on the unit square with size

N = 11, 12, ..., 100. For Euclidean instances in the second row of both tables, consider a set

of Euclidean instances with size N = 11, 12, ..., 100 and x-y coordinates of vertices that have

either uniform, normal, or arcsine distributions. In this instance set containing 270 instances, the

structural properties differ among instances. We can still observe relatively high r-squared values

and low MAPE values when using stdST to predict T .

       We also experiment with non-Euclidean instances. First, we construct a set of instances

with N = 30, 40, ..., 100 vertices uniformly distributed on a unit square and apply the non-

Euclidean distance by random multiplication for ϵ = 0.1, 0.2, ..., 0.9. Next, we construct a set

of instances with N = 30, 40, ..., 100 vertices uniformly distributed on a unit square and apply

the non-Euclidean distance by random substitution for θ = 0.1, 0.2, ..., 0.9. Finally, we construct



                                                144


--- PAGE BREAK ---

                                 Variable Coefficients
                                                             stdST      constant        Adj.R2     MAPE
 Instance Set
                        Uniform                            1.055∗∗∗      1.003∗∗∗       0.922       8.130%
                       Euclidean                           0.848∗∗∗      1.062∗∗∗       0.952       9.348%
                     Multiplication                       −0.065         1.443∗∗∗       0.002      30.348%
                      Substitution                         0.949∗∗∗      0.911∗∗∗       0.802       7.214%
                        Joint Set                          0.162         1.344∗∗∗       0.018      24.314%
                    Normally Distance                     −1.710∗∗∗     16.512∗∗∗       0.887      21.733%
 ***
       p < 0.01, ** p < 0.05, * p < 0.1

                                 Table B.1: The stdST Model on MST Instances


                            Variable Coefficients
                                                    meanST      stdST      constant       Adj.R2    MAPE
 Instance Set
                     Uniform                         0.054∗∗∗   1.327∗∗∗     1.057∗∗∗      0.948     7.264%
                                                          ∗∗∗        ∗∗∗
                    Euclidean                        0.030      1.657        0.816∗∗∗      0.946     9.945%
                                                          ∗∗∗        ∗∗∗
                  Multiplication                     0.161     −2.666        5.173∗∗∗      0.931     7.842%
                                                          ∗∗∗
                   Substitution                      0.085     −0.508        2.927∗∗∗      0.831     5.819%
                                                          ∗∗∗        ∗∗∗
                     Joint Set                       0.138     −2.143        4.543∗∗∗      0.861     8.529%
                                                                     ∗∗∗
                 Normally Distance                   1.508    −17.905    −1198.605∗∗∗      0.969    15.464%
 ***
       p < 0.01, ** p < 0.05, * p < 0.1

                        Table B.2: The meanST and stdST Models on MST Instances



a joint set that contains instances from the two previous sets. In both Tables B.1 and B.2, these

are denoted by Multiplication, Substitution, and a Joint Set respectively. According to Table B.1,

for the random multiplication set, the performance of stdST in predicting T is extremely poor.

On the random substitution set, we observe a moderate to strong power-law relationship between

stdST and T . Not surprisingly, the result for the combined set is in between.

          The previous experiments still consider instances with some metric structure. We also ex-

periment on instances for which we do not locate vertices in an x-y coordinate system; rather, we

generate non-diagonal entries in the distance matrix according to a normal distribution N (100, α).

We generate 50 instances in this set with size N = 50 and α ranging from 0, 1, ..., 49. The results

for this set are provided in the last row of the two tables. For these instances without any geo-



                                                         145


--- PAGE BREAK ---

metric interpretation, the standard deviation predictor still provides reasonable predictive results.

Combining the last two sets of experiments on non-Euclidean instances, we observe that the pre-

dictive result of stdST is generally not as good as it is for Euclidean instances. In addition, stdST

can be a reasonable predictor for some sets, but not for all.

      Based on Tables B.1 and B.2, we observe that adding the meanSP predictor to the model

can enhance predictive ability most of the time. The results are more obvious for the multiplica-

tion set and the joint set, for which the relationship between stdST and T is barely observable.

For the Euclidean instances in the first and second rows, using the model from Equation (B.3)

provides results either better than or comparable to those obtained using the model from Equation

(B.1). According to the results from Tables B.1 and B.2, for the MST, we can observe a strong

correlation between the feasible solution values and optimal solution value.



B.2    Results for the MWM

      Similar to our experiments on MST instances, we test the predictive ability of stdM on

M and the combination of meanM and stdM on M in different sets of MWM instances. As

mentioned earlier, we use linear regression to generate the constants in Equations (B.2) and (B.4)

on different instance sets. The fitted coefficients and results of predicting M using only the stdM

predictor are summarized in Table B.3. The fitted coefficients and results of the model including

both meanM and stdM are shown in Table B.4.

      For each of the following instance sets, we generate each set of instances with size N =

20, 22, ..., 100 for the purpose of generating perfect matchings. We first consider Euclidean in-

stances with vertices either uniformly distributed on a unit square or with a bivariate normal



                                                146


--- PAGE BREAK ---

                                Variable Coefficients
                                                            stdM          constant          Adj.R2     MAPE
 Instance Set
                       Euclidean                            1.084∗∗∗          2.983∗∗∗      0.901      18.867%
                      Multiplication                        3.002∗∗∗          1.007∗∗∗      0.854      21.661%
                      Substitution                          0.181∗∗∗          2.112∗∗∗      0.290      24.845%
                       Joint Set                            0.870∗∗∗          2.916∗∗∗      0.294      25.099%
 ***
       p < 0.01, ** p < 0.05, * p < 0.1

                                Table B.3: The stdM Model on MWM Instances


                            Variable Coefficients
                                                    meanM          stdM         constant      Adj.R2    MAPE
 Instance Set
                     Euclidean                      1.450∗∗∗       0.345∗∗      −0.451∗∗∗     1.000     1.417%
                    Multiplication                  1.324∗∗∗       2.446∗∗∗     −1.245∗∗∗     0.999     2.508%
                    Substitution                    0.986∗∗∗       1.314∗∗∗      0.153        0.977     3.768%
                     Joint Set                      0.993∗∗∗       1.202∗∗∗      0.153        0.979     3.622%
 ***
       p < 0.01, ** p < 0.05, * p < 0.1

                       Table B.4: The meanM and stdM Models of MWM Instances


                                           
               0.5 0.01 0 
distribution N 
                 , 
                                   . These would be the instances in the Euclidean set in
                                      
                   0.5       0 0.01
row 1 of Tables B.3 and B.4, and, as we observe, there is a strong power-law relationship between

stdM and M in this set of Euclidean instances. However, the adjusted r-squared value is almost

1 when using both meanM and stdM to estimate M .

          Using the two definitions of non-Euclidean distances established with respect to the TSP,

on unit squares, we construct a set of instances with non-Euclidean distances by random mul-

tiplication for ϵ = 0.1, 0.3, ..., 0.9, a set of instances with non-Euclidean distances by random

substitution for θ = 0.1, 0.3, ..., 0.9, and a joint set containing instances from the two previous

sets. In Tables B.3 and B.4, these sets are called Multiplication, Substitution, and Joint Set,

respectively.

          According to Table B.3, for the random substitution set, the performance of stdM in pre-



                                                        147


--- PAGE BREAK ---

dicting M is relatively poor. In the random multiplication set, we observe a moderate to strong

power-law relationship between stdM and M . Also, while the result for the combined set is in

between, we find that, in general, stdM can sometimes, but not always, do well as a predictor of

M.




                                              148


--- PAGE BREAK ---

Appendix C:       Detailed SDVRP Instance Generation



      In this appendix, we detail how we generate SDVRP instances using a structured random-

ization approach. Each instance is defined by parameters controlling the number of customers,

depot location, customer geographic distribution, demand type, and average route size. Each

instance is characterized by:


    • n: The number of customers.


    • Depot Positioning: Controls where the depot is placed:


         – Randomly positioned within a rectangular region.

         – Centrally positioned in the rectangular region.

         – Fixed at coordinates (0,0).


    • Customer Positioning: Determines how customers are distributed:


         – All customers are uniformly distributed within the rectangular region.

         – All customers are clustered. The number of cluster centers is randomly drawn from

           U {2, 6}, with each center uniformly placed in the region. Additional customers are

           assigned to clusters using an accept-reject sampling method, ensuring their locations

           follow a decaying probability function relative to the cluster centers.


                                               149


--- PAGE BREAK ---

         – A mix of customer distributions, with half placed randomly and half assigned to clus-

           ters.


    • Demand Type: Defines the distribution of customer demands:


         – Demand values are drawn from a uniform distribution U {1, 100}.

         – Demand values are drawn from a uniform distribution U {50, 100}.


    • Average Route Size: Specifies the expected number of routes, which is used to determine

      the vehicle capacity for the instance.


      To ensure a diverse dataset, we systematically iterate over different parameter combina-

tions, generating a comprehensive set of instances for evaluation.




                                               150


--- PAGE BREAK ---

References


Akkerman, F., & Mes, M. (2022). Distance approximation to support customer selection in
      vehicle routing problems. Annals of operations research, 1–29.
Akkerman, F., Mes, M., & Heijnen, W. (2020). Distance approximation for dynamic waste
      collection planning. In Computational Logistics (pp. 356–370).
Applegate, D. L., Bixby, R. E., Chvátal, V., & Cook, W. J. (2011). The traveling salesman
      problem. Princeton University Press.
Archetti, C., Savelsbergh, M. W., & Speranza, M. G. (2006). Worst-case Analysis for Split
      Delivery Vehicle Routing Problems. Transportation Science, 40(2), 226–234.
Archetti, C., & Speranza, M. G. (2008). The Split Delivery Vehicle Routing Problem: A Sur-
      vey. In B. Golden, S. Raghavan, & E. Wasil (Eds.), The Vehicle Routing Problem: Latest
      Advances and New Challenges (pp. 103–122). Springer.
Basel, J., & Willemain, T. R. (2001). Random tours in the traveling salesman problem: Analysis
      and application. Computational Optimization and Applications, 20(2), 211–217.
Beardwood, J., Halton, J. H., & Hammersley, J. M. (1959). The shortest path through many
      points. In Mathematical Proceedings of the Cambridge Philosophical Society (Vol. 55, pp.
      299–327).
Bertazzi, L., & Wang, X. (2022). Matheuristics with Performance Guarantee for the Unsplit and
      Split Delivery Capacitated Vehicle Routing Problem. Networks, 80(4), 482–501.
Boeing, G. (2017). Osmnx: New methods for acquiring, constructing, analyzing, and visualizing
      complex street networks. Computers, Environment and Urban Systems, 65, 126–139.
Breiman, L. (2001). Random forests. Machine learning, 45, 5–32.
Cavdar, B., & Sokol, J. (2015). A Distribution-free TSP Tour Length Estimation Model for
      Random Graphs. European Journal of Operational Research, 243(2), 588–598.
Chazelle, B. (2000). A Minimum Spanning Tree Algorithm with Inverse-Ackermann Type
      Complexity. Journal of the ACM (JACM), 47(6), 1028–1047.
Chien, T. W. (1992). Operational estimators for the length of a traveling salesman tour. Comput-
      ers & Operations Research, 19(6), 469–478.
Christofides, N., & Eilon, S. (1969). Expected Distances in Distribution Problems. Journal of
      the Operational Research Society, 20(4), 437–443.
Christofides, N., Mingozzi, A., & Toth, P. (1979). The Vehicle Routing Problem. Combinatorial
      Optimization, John Wiley and Sons, London.
Clarke, G., & Wright, J. W. (1964). Scheduling of Vehicles from a Central Depot to a Number
      of Delivery Points. Operations Research, 12(4), 568–581.
Croes, G. A. (1958). A Method for Solving Traveling-Salesman Problems. Operations Research,
      6(6), 791–812.
Daganzo, C. F. (1984). The length of tours in zones of different shapes. Transportation Research


                                              151


--- PAGE BREAK ---

      Part B: Methodological, 18(2), 135–145.
Dror, M., & Trudeau, P. (1989). Savings by Split Delivery Routing. Transportation Science,
      23(2), 141–145.
Duan, R., & Pettie, S. (2014). Linear-time Approximation for Maximum Weight Matching.
      Journal of the ACM (JACM), 61(1), 1–23.
Eilon, S., Watson-Gandy, C. D. T., Christofides, N., & de Neufville, R. (1974). Distribution
      Management-Mathematical Modelling and Practical Analysis. IEEE Transactions on Sys-
      tems, Man, and Cybernetics(6), 589–589.
Figliozzi, M. A. (2008). Planning Approximations to the Average Length of Vehicle Routing
      Problems with Varying Customer Demands and Routing Constraints. Transportation Re-
      search Record, 2089(1), 1–8.
Forrest, J., & Lougee-Heimer, R. (2005). CBC User Guide. In Emerging Theory, Methods, and
      Applications (pp. 257–277). INFORMS.
Gamst, M., Lusby, R. M., & Ropke, S. (2024). Exact and heuristic methods for the split delivery
      vehicle routing problem. Transportation Science, 58(4), 741–760.
Hagberg, A., Swart, P., & S Chult, D. (2008). Exploring Network Structure, Dynamics, and
      Function using NetworkX (Tech. Rep.). Los Alamos National Lab.(LANL), Los Alamos,
      NM (United States).
Haimovich, M., & Rinnooy Kan, A. H. (1985). Bounds and Heuristics for Capacitated Routing
      Problems. Mathematics of Operations Research, 10(4), 527–542.
Har-Peled, S. (2011). On the expected complexity of random convex hulls. arXiv preprint
      arXiv:1111.5340.
Helsgaun, K. (2000). An effective implementation of the lin–kernighan traveling salesman
      heuristic. European Journal of Operational Research, 126(1), 106–130.
Hoffman, K. L., Padberg, M., Rinaldi, G., et al. (2013). Traveling salesman problem. Encyclo-
      pedia of Operations Research and Management Science, 1, 1573–1578.
Hougardy, S., & Zhong, X.            (2018).      The Tn,m Instances in the TSPLIB For-
      mat. Retrieved 2018-09, from http://www.or.uni-bonn.de/˜hougardy/
      HardTSPInstances.html
Hougardy, S., & Zhong, X. (2021). Hard to solve instances of the euclidean traveling salesman
      problem. Mathematical Programming Computation, 13(1), 51–74.
Kendall, W. S., Last, G., & Molchanov, I. S. (2009). New perspectives in stochastic geometry.
      Oberwolfach Reports, 5(4), 2655–2702.
Kool, W., Van Hoof, H., & Welling, M. (2018). Attention, Learn to Solve Routing Problems!
      arXiv preprint arXiv:1803.08475.
Kou, S., Golden, B., & Bertazzi, L. (2024a). Estimating optimal split delivery vehicle routing
      problem solution values. Computers & Operations Research, 168, 106714.
Kou, S., Golden, B., & Bertazzi, L. (2024b). An improved model for estimating optimal vrp
      solution values. Optimization Letters, 18(3), 697–703.
Kou, S., Golden, B., & Bertazzi, L. (2025). Machine Learning Models to Optimal TSP Tour
      Length Estimation. Manuscript submitted for publication.
Kou, S., Golden, B., & Poikonen, S. (2022). Optimal TSP tour length estimation using standard
      deviation as a predictor. Computers & Operations Research, 148, 105993.
Kou, S., Golden, B., & Poikonen, S. (2023). Optimal TSP tour length estimation using Sammon
      maps. Optimization Letters, 17(1), 89–105.

                                             152


--- PAGE BREAK ---

Kou, S., Golden, B., & Poikonen, S. (2024). Estimating optimal objective values for the TSP,
       VRP, and other combinatorial problems using randomization. International Transactions
       in Operational Research, 31(5), 3443–3458.
Kwon, O., Golden, B., & Wasil, E. (1995). Estimating the Length of the Optimal TSP tour:
       An Empirical Study using Regression and Neural Networks. Computers & Operations
       Research, 22(10), 1039–1046.
Laporte, G., & Palekar, U. (2002). Some applications of the clustered travelling salesman prob-
       lem. Journal of the Operational Research Society, 53(9), 972–976.
Lellouche, S., & Souris, M. (2020). Distribution of distances between elements in a compact set.
       Stats, 3(1), 1–15.
Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., & et al. (2024). Kan: Kolmogorov-Arnold
       Networks. arXiv preprint arXiv:2404.19756.
Merchán, D., & Winkenbach, M. (2019). An empirical validation and data-driven extension
       of continuum approximation approaches for urban route distances. Networks, 73(4), 418–
       433.
Mitchell, S., OSullivan, M., & Dunning, I. (2011). Pulp: A Linear Programming Toolkit for
       Python. The University of Auckland, Auckland, New Zealand, 65.
Nicola, D., Vetschera, R., & Dragomir, A. (2019). Total distance approximations for routing
       solutions. Computers & Operations Research, 102, 67–74.
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., . . . others (2011).
       Scikit-learn: Machine learning in python. The Journal of Machine Learning Research, 12,
       2825–2830.
Pekalska, E., de Ridder, D., Duin, R. P., & Kraaijveld, M. A. (1999). A New Method of Gen-
       eralizing Sammon Mapping with Application to Algorithm Speed-up. In Heijen nl (pp.
       221–228).
Pettie, S., & Ramachandran, V. (2002). An Optimal Minimum Spanning Tree Algorithm. Journal
       of the ACM (JACM), 49(1), 16–34.
Philip, J. (2007). The probability distribution of the distance between two random points in a
       box. KTH Mathematics, Royal Institute of Technology.
Queiroga, E., Sadykov, R., Uchoa, E., & Vidal, T. (2021). 10,000 Optimal CVRP Solutions for
       Testing Machine Learning Based Heuristics. In AAAI-22 Workshop on Machine Learning
       for Operations Research (ML4OR).
Reinelt, G. (1991). TSPLIB—A Traveling Salesman Problem Library. ORSA Journal on Com-
       puting, 3(4), 376–384.
Robust, F., Daganzo, C. F., & Souleyrette II, R. R. (1990). Implementing vehicle routing models.
       Transportation Research Part B: Methodological, 24(4), 263–286.
Sammon, J. W. (1969). A nonlinear mapping for data structure analysis. IEEE Transactions on
       computers, 100(5), 401–409.
Silva, M. M., Subramanian, A., & Ochi, L. S. (2015). An iterated local search heuristic for the
       split delivery vehicle routing problem. Computers & Operations Research, 53, 234–249.
Sinha Roy, D., Golden, B., Masone, A., & Wasil, E. (2023). Using Regression Models to Un-
       derstand the Impact of Route-length Variability in Practical Vehicle Routing. Optimization
       Letters, 17(1), 163–175.
Sobhanan, A., Park, J., Park, J., & Kwon, C. (2024). Genetic Algorithms with Neural Cost
       Predictor for Solving Hierarchical Vehicle Routing Problems. Transportation Science.

                                                153


--- PAGE BREAK ---

Sutcliffe, P. J., Solomon, A., & Edwards, J. (2012). Computing the Variance of Tour Costs
      over the Solution Space of the TSP in Polynomial Time. Computational Optimization and
      Applications, 53(3), 711–728.
Uchoa, E., Pecin, D., Pessoa, A., Poggi, M., Vidal, T., & Subramanian, A. (2017). New bench-
      mark instances for the capacitated vehicle routing problem. European Journal of Opera-
      tional Research, 257(3), 845-858.
Vakhutinsky, A. I., & Golden, B. L. (1995). A Hierarchical Strategy for Solving Traveling
      Salesman Problems using Elastic Nets. Journal of Heuristics, 1, 67–76.
Varol, T., Özener, O. Ö., & Albey, E. (2024). Neural Network Estimators for Optimal Tour
      Lengths of TSP Instances with Arbitrary Node Distributions. Transportation Science,
      58(1), 45–66.
Vinel, A., & Silva, D. F. (2018). Probability distribution of the length of the shortest tour between
      a few random points: a simulation study. In 2018 Winter Simulation Conference (WSC)
      (pp. 3156–3167).
Vinyals, O., Fortunato, M., & Jaitly, N. (2015). Pointer Networks. Advances in Neural Informa-
      tion Processing Systems, 28.
Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., . . . others
      (2020). Scipy 1.0: Fundamental algorithms for scientific computing in python. Nature
      Methods, 17(3), 261–272.
Wang, C. J., Fang, H., & Wang, H. (2016). ESammon: A Computationaly Enhanced Sam-
      mon Mapping Based on Data Density. In 2016 international conference on computing,
      networking and communications (icnc) (pp. 1–5).
Webb, M. (1968). Cost Functions in the Location of Depots for Multiple-Delivery Journeys.
      Journal of the Operational Research Society, 19, 311–320.
Yang, H., Lu, X., Zhou, X., Zheng, R., Liu, Y., & Tao, S. (2022). Expected Length of the Shortest
      Path of the Traveling Salesman Problem in 3D Space. Journal of Advanced Transportation,
      2022.




                                                154


--- PAGE BREAK ---


