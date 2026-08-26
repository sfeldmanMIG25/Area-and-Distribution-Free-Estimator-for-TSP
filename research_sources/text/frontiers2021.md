# Solving the Large-Scale {TSP} Problem in 1 h: {Santa Claus Challenge 2020}

> Citation key: `frontiers2021`  
> DOI: 10.3389/frobt.2021.689908  
> Download URL: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.689908/pdf  
> SHA-256: `ea39be2e13ae94c78dbb10fd45ef3ba8221d796d211578d2d609e4832dd94237`  

---

                                                                                                                                        ORIGINAL RESEARCH
                                                                                                                                    published: 04 October 2021
                                                                                                                                doi: 10.3389/frobt.2021.689908




                                            Solving the Large-Scale TSP Problem
                                            in 1h: Santa Claus Challenge 2020
                                            Radu Mariescu-Istodor and Pasi Fränti *

                                            School of Computing, University of Eastern Finland, Joensuu, Finland


                                            The scalability of traveling salesperson problem (TSP) algorithms for handling large-scale
                                            problem instances has been an open problem for a long time. We arranged a so-called
                                            Santa Claus challenge and invited people to submit their algorithms to solve a TSP problem
                                            instance that is larger than 1 M nodes given only 1 h of computing time. In this article, we
                                            analyze the results and show which design choices are decisive in providing the best
                                            solution to the problem with the given constraints. There were three valid submissions, all
                                            based on local search, including k-opt up to k  5. The most important design choice
                                            turned out to be the localization of the operator using a neighborhood graph. The divide-
                                            and-merge strategy suffers a 2% loss of quality. However, via parallelization, the result can
                                            be obtained within less than 2 min, which can make a key difference in real-life applications.
                                            Keywords: GRID, graphs, clustering, divide and conquer, TSP, scalability



                                            INTRODUCTION
                           Edited by:       The traveling salesperson problem (TSP) (Applegate et al., 2011) is a classical optimization problem
                Sheri Marina Markose,       that aims to ﬁnd the shortest path that connects a given number of cities. Even though the name
 University of Essex, United Kingdom        implies outdoor movement by human beings, there are many other applications of TSP, including
                       Reviewed by:         planning, scheduling of calls, manufacturing of microchips, routing of trucks, and parcel delivery.
                     Marco Gavanelli,          TSP is not limited to physical distances, but any other optimization function in place of the
            University of Ferrara, Italy    distance can be used. In the early days, travel distance or travel time were used (Dantzig and Ramser,
             Konstantinos Giannakis,        1959), but currently, other objectives also exist, such as minimizing the exposure to sunlight (Li et al.,
        University of Bergen, Norway
                                            2019), maximizing the safety of travel (Krumm and Horvitz, 2017), or minimizing CO2 levels (Kuo
                  *Correspondence:          and Wang, 2011; Lin et al., 2014). Practical applications include logistics such as vacation planning or
                           Pasi Fränti      orienteering (Fränti et al., 2017), transportation of goods, providing maintenance, or offering health
                       franti@cs.uef.ﬁ      services (Golden et al., 2008), among others.
                                               Sometimes there is a need to process larger problem instances, especially in applications such as
                  Specialty section:        waste collection and delivery of goods, where millions of deliveries occur each day. Due to the
       This article was submitted to        COVID-19 pandemic, logistics became the center of attention, causing couriers to struggle to meet
Computational Intelligence in Robotics,     increased demands from hospitals, supermarkets, and those who self-isolate at home and order food
              a section of the journal
                                            and merchandise online.
        Frontiers in Robotics and AI
                                               TSP is an NP-hard problem. Despite this fact, there are algorithms that can solve impressively
           Received: 01 April 2021
                                            large instances. Some of these are exact solvers that guarantee the optimum solution (Barnhart et al.,
         Accepted: 17 August 2021
                                            1998; Applegate et al., 1999; Bixby, 2007). However, these methods have exponential time complexity
        Published: 04 October 2021
                                            and become impractical on large-scale problem instances. The current record includes 85,900 targets
                              Citation:
                                            that were solved in approximately 136 CPU-years (Applegate et al., 2011). For this reason, heuristic
Mariescu-Istodor R and Fränti P (2021)
Solving the Large-Scale TSP Problem
                                            solvers have been developed, and they currently outnumber the exact solution methods.
 in 1 h: Santa Claus Challenge 2020.           Existing heuristics use different strategies, such as local search, tabu search, ant colony
            Front. Robot. AI 8:689908.      optimization, simulated annealing, and evolutionary methods (Braekers et al., 2016). These
      doi: 10.3389/frobt.2021.689908        methods can quickly provide suboptimal solutions that are acceptable in practice. For example,


Frontiers in Robotics and AI | www.frontiersin.org                                 1                                   October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                                             Santa Claus Challenge 2020



we can solve an instance with 1,904,711 targets within 15% of the                       To address this problem, we created a TSP challenge where
optimum in 30 min and within 0.1% of the optimum in a few                            participants were asked to provide an algorithm to optimize Santa
hours (Helsgaun, 2000; Taillard and Helsgaun, 2019). The quality                     Claus tours to visit all households in Finland. The algorithms
of the solution increases with the processing time the algorithm                     should terminate within 1 h. In this article, we perform an
spends.                                                                              analytical comparison of the submitted algorithms. We analyze
    A generalized variant of TSP is the vehicle routing problem                      each design component separately, including the single-solution
(VRP) (Dantzig and Ramser, 1959). This problem considers                             TSP solver, clustering method for dividing the problem into
several real-world elements, such as multiple agents (k-TSP),                        subproblems, merge strategy to combine the subsolutions,
limited capacity, working hours, time windows, and depots. VRP                       approaches to the open-loop, ﬁxed start point, and k-TSP cases.
instances are more difﬁcult to solve, and most research considers
instances with only a few hundred targets (Dumas et al., 1991;                       Data and Goals
Reinelt, 1992; Uchoa et al., 2017). In fact, the ﬁrst instances with                 In the spirit of Christmas, Santa Claus needs to deliver presents to
up to 30,000 targets appeared just last year (Arnold et al., 2019),                  the children in every family on Christmas Eve. He can also use k
even though there is a real demand for even much higher                              helpers by dividing the tour into k parts accordingly. We awaited
instances. FedEx, for example, have over six million deliveries                      solutions to the following problem cases:
per day.1
    When solving large-scale problem instances, the scalability of                   1. Closed-loop TSP;
the algorithms is still an issue. One issue is that the many                         2. Open-loop TSP;
algorithms are not linear in time and therefore not scalable.                        3. Fixed-start TSP (open loop);
Another issue is that many implementations use a complete                            4. Multiple tours k-TSP (open loop).
distance matrix that cannot ﬁt into memory with large-scale
instances. One solution is to split the problem into subproblems                        The ﬁrst case is the classical (closed-loop) TSP problem where
and use multiple cores. However, NP-hard problems cannot be                          Santa needs to return home to complete the tour. The three other
naturally divided without compromising the quality. It is not                        cases are open-loop variants that create a TSP path where return
trivial to create a good split-and-merge strategy that would keep                    to home is not necessary. In the second case, Santa can start from
the loss of quality marginal. All of these issues pose challenges to                 any location with the logic that he has plenty of time to prepare
applications of TSP to large-scale problem instances.                                and can go to the selected start point well before the actual trip
    The concept of large-scale itself has evolved considerably over                  starts; only the time spent traveling the path counts. In the third
the years. In a study by Thibaut and Jaszkiewicz (2010), problem                     case, he leaves from his home (depot), which in our data is set to
sizes up to N  1,000 were considered. Sakurai et al. (2006)                         Rovaniemi. The fourth case is motivated by the fact that it would
performed experiments with only N  40 targets, and the problem                      be impossible for Santa to make the trip on Christmas Eve
instances of N  1,000–10,000 were called very large scale. One                      without breaking the laws of physics.2 Santa therefore recruits
obvious approach to attack the issue is to divide the problem into                   k assistants; elves or drones in modern times (Poikonen et al.,
smaller subproblems. Different space partitioning methods have                       2019) and divides the tour into multiple parts that are solved by
been used, such as Karp, Strip, and Wedging insertion                                each helper independently.
(Valenzuela and Jones, 1995; Xiang et al., 2015). The afﬁnity
propagation clustering algorithm was used by Jiang et al. (2014)                     Data
for problems of size N < 3,000 and hierarchical k-means for                          We extracted all building locations in Finland from
problems of size N > 3,000. Each cluster had 40 targets at the                       OpenStreetMap3 (OSM) to create 1,437,195 targets that
most. However, the reported results took 3 days to compute                           represent the households for Santa to visit. We used our local
problem sizes over 1 M.                                                              installation of the OSM and the Overpass API. We ﬁrst queried
    When using clustering, it is also possible that some clusters are                inside a bounding box that encompasses the entire country and
larger than others, which could lose the beneﬁt of clustering. An                    kept only the buildings that fell within the country borders. The
attempt to reach balanced cluster sizes was considered by Yang                       coordinates were converted into the Finnish National Coordinate
et al. (2009). Another problem is memory consumption because                         System (KKJ), which projects them in Euclidean space (most
many algorithms store an N×N distance matrix. Additional                             typical for researchers), where the Euclidean distance
memory is needed in some cases, such as storing the                                  corresponds to meters of movement in a straight line. The
pheromone trails in the case of ant colony optimization.                             data are published as a TSP instance on our website.4
Nevertheless, a version of ant colony optimization by Chitty                            We note that the building locations do not match 1:1 to the
(2017) was capable of handling these issues and solving problem                      households in Finland, and there is bias. Some regions,
sizes up to 200 k at the cost of a 7% decrease in accuracy. This cost                especially in the southeastern part, have denser records of
(decrease) in the performance is quite large to pay for the                          buildings than the other areas. This arrangement shows
scalability.

                                                                                     2
                                                                                       https://gizmodo.com/can-santa-claus-exist-a-scientiﬁc-debate-1669957032
1                                                                                    3
 https://www.statista.com/statistics/878354/fedex-express-total-average-daily-         https://www.openstreetmap.org
                                                                                     4
packages                                                                               http://cs.uef.ﬁ/sipu/santa/data.html



Frontiers in Robotics and AI | www.frontiersin.org                               2                                         October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                                                       Santa Claus Challenge 2020




  FIGURE 1 | The Santa Claus TSP challenge was motivated by Santa’s need to deliver presents to the children in every family on Christmas Eve, as is the tradition in
  Finland. We used publicly available data from OSM, consisting of 1.4 million building locations. While most of the population is in south Finland, this data set has dense
  coverage of certain regions, such as southeast Finland. We note that this database does not cover the entirety of Finland, and there is a bias in the locations.




visible artifacts as sharp edges between the dense and sparse                              SOLVING LARGE-SCALE PROBLEMS
regions (see Figure 1). This appearance is most likely due to
an inconsistent representation in the OSM database of north                                The general structure of all methods discussed here follows the same
vs. south. The data itself appear to be correct, but there is a                            overall structure, which consists of the following components:
lack of data in many regions. However, for the purpose of our
Santa competition, the resulting data set is good enough,                                     • Single-solution TSP solver,
because the size of the data serves the need for a large-scale                                • Divide into smaller subproblems,
test case (n ≈ 1.4 M).                                                                        • Merge the subsolutions.

Goals
We set a requirement that a submitted algorithm must optimize                              Single-Solution TSP Solver
the tour within 1 h. All of the valid submissions were evaluated                           State-of-the-art TSP solvers (excluding the optimal ones) and all
after executing for 1 h or less. A program was disqualiﬁed if it did                       submitted algorithms are based on local search. The idea is to
not terminate within 1 h. All of the submissions were evaluated in                         have an initial solution that is then improved by a sequence of
terms of quality, speed, and simplicity. We ran all of the                                 local operators in a trial-and-error manner. The key component
algorithms on the same Dell R920 machine with 4 × E7-4860                                  in the local search is the choice of the operator. It deﬁnes how the
(a total of 48 cores), 1 TB, and 4 TB SAS HD.                                              current solution is modiﬁed to generate new candidate solutions.
   While Santa has time during the entire year for his                                     The following operators were used:
preparations, we set a stricter requirement. We allowed the
computer program to spend only 1 h performing the necessary                                   • Relocate (Gendreau et al., 1992),
calculations. This approach reﬂects more real-life applications                               • Link swap (Sengupta et al., 2019),
where the situation is dynamic and constantly changing. The                                   • 2-opt (Croes, 1958),
purpose is to test the scalability of algorithms for real-life                                • k-opt (Shen and Kernighan, 1973).
applications.
                                                                                               The most popular of these is 2-opt (Croes, 1958) and its
Challenges                                                                                 generalized variant k-opt (Shen and Kernighan, 1973). The 2-opt
The greatest challenge is the size of the data and the limited                             operator selects two links and redirects the links between their
processing time of 1 h. Most TSP solvers have quadratic or higher                          nodes. Its generalized variant, k-opt, involves k links, which
time complexity, and even simple greedy heuristics need to                                 allows more complex modiﬁcations of the solution. It is also
compute the distance to all remaining targets at each of the n                             known as the Lin–Kernighan heuristic. A Link swap relocates any
steps (Fränti et al., 2021). Even such algorithms are not fast                             link by connecting it to one or both end points of the current path.
enough to complete in 1 h on the speciﬁed machine.                                         Link swap works only for the open-loop case but has been found
   Another challenge is the multiple variants. Most existing                               to contribute most to the search; approximately 50% of the
methods are tailored for the classical closed-loop case, and                               improvements were due to Link swap as shown by Sengupta
some modiﬁcations are needed to adapt them to the open-                                    et al. (2019). Relocate removes a node from its current position in
loop, ﬁxed-start, and k-TSP cases.                                                         the path and reallocates it elsewhere by creating a new detour via


Frontiers in Robotics and AI | www.frontiersin.org                                     3                                            October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                                  Santa Claus Challenge 2020




  FIGURE 2 | Local search operators used in the submitted algorithms.




this node. The operators are demonstrated in Figure 2. Tilo                      geographic data; a linear time solution for the Delaunay graph can
Strutz pointed out that both Relocate and Link swap are special                  be found in the work by Peterson (1998).
cases of k-opt (Strutz, 2021).                                                       The drawback of localization is that successful operations can
    Most local search algorithms apply one of the three strategies:              be missed due to arbitrary grid divisions or limitations in the
random, best improvement, and prioritization. Random search                      neighborhood graphs. However, it was shown by Fränti et al.
tries out the operators in random order and accepts any solution                 (2016) that 97% of the links in the optimal TSP path are included
that improves the current solution. Best improvement considers all               in the XNN graph. Localization is an efﬁcient approach to reduce
possible operators for the current solution and selects the best. The            unnecessary calculations and is expected not to miss many of the
operators and their parameters can also be prioritized based on                  moves that full search would potentially ﬁnd.
criteria such as alpha-nearness in an effective implementation of
the Lin–Kernighan heuristic known as LKH (Helsgaun, 2000).                       Divide by Clustering
Here, each link is scored based on sensitivity analysis using                    To make the algorithm scalable, we expect that it is necessary to
minimum spanning trees (MST). More speciﬁcally, it measures                      divide the data into smaller size instances that the TSP solver can
the increase in cost when a tree is forced to contain the speciﬁc link.          manage. Spatial clustering is applied here. The idea is that each
                                                                                 subproblem is solved separately, and the resulting paths are then
Localizing the Search Space                                                      merged to create the overall TSP path.
One limitation of the above-mentioned operators is that most of                     To select the most appropriate clustering method, we must
the generated candidates are meaningless. For example, it makes                  consider three questions: 1) which objective function to optimize,
no sense to try to relocate a node along a path at the opposite part             2) which algorithm to perform this optimization, and 3) how
of the space. In most cases, only the nearby points are meaningful               many clusters. Assume that the TSP solver requires quadratic,
(see Figure 3). In the case of relocation and 2-opt, only a few out              O(N2), time complexity. If the data are equally divided into √N
of the n − 2 choices have a realistic chance of success, and most                clusters, we would have √N points in each cluster. Solving TSP
others would be just a waste of time. Link swap is somewhat better               for one cluster would require O(N) time and O(N1.5) for all √N
because there are only three choices, of which at least the one                  clusters. With our data (≈1.4 M points), we would have
where both end points are changed is potentially meaningful.                     approximately 1,200 clusters. Multithreading with >1,200
    Two strategies have been applied to restrict the operators to                processors would make the computation time linear.
consider targets only in the local neighborhood: grid and                           However, it is not clear how the clusters should be optimized.
neighborhood graph. The ﬁrst strategy is to divide the space into                Figure 5 shows several possible methods. K-means minimizes the
cells by generating a grid (see Figure 4). The operator is then restricted       sum-of-squared errors and generates spherical clusters. This
to considering only nodes and links within the same cell or targets in           approach is a reasonable clustering strategy in general but not
neighboring cells. Because the density of data varies considerably and           necessarily the optimal strategy for TSP. Grid-based methods are
the goal is to limit the number of nodes per cell, multiple resolutions          faster but even worse, in general, because the borders are decided
are often necessary, and smaller cells are generated only when needed.           without considering any properties of the data. On the other
    The second strategy is to create a neighborhood graph such as                hand, the TSP path is also a spanning tree, and the clusters
the Delaunay graph or related graphs such as Gabriel or XNN.                     correspond to a spanning forest. A single-link clustering
Efﬁcient algorithms exist to compute these data structures for 2D                algorithm might therefore make sense because it ﬁnds a


Frontiers in Robotics and AI | www.frontiersin.org                           4                                   October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                                                     Santa Claus Challenge 2020




  FIGURE 3 | Local search operators should consider only nearby points and links. A given target has n − 2  13 new positions, where it could be relocated. However,
  only the nearby points (marked by arrows) are worthwhile considering.




minimum spanning forest. However, when the density varies,                                Merging the Subsolutions
large clusters will form in regions with high density (southern                           Another open question in the algorithms is how to merge the
Finland), which are too large to be optimized properly. Attempts                          subtours from the individual clusters into a single tour. All divide
to mitigate this behavior were tried out by Mariescu-Istodor et al.                       and conquer submissions decide the merging strategy before
(2021), but those approaches were too slow to handle more than a                          solving the clusters. The problem is essentially formulated as a
few tens of thousands of targets efﬁciently. The relationship                             cluster-level TSP, where each cluster represents a node in a so-
between MST and TSP was also utilized by Fränti et al.                                    called meta-level graph. This meta-level problem is solved to
(2021). Density peaks clustering was considered by Liao and                               decide which clusters are to be merged (Kobayashi, 1998). The
Liu (2018).                                                                               subproblems within each cluster are treated as open-loop TSP



Frontiers in Robotics and AI | www.frontiersin.org                                    5                                           October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                                                       Santa Claus Challenge 2020




  FIGURE 4 | Efﬁciency of the local search operators depends highly on how to limit the search for candidates within the neighborhood. The grid-based approach
  restricts the two operands (links) to be in the same or neighboring grid cells, while a neighborhood graph considers only connected nodes.




  FIGURE 5 | An example of a TSP instance divided into clusters in three different ways. The optimum cannot be achieved when it enters the same cluster more than once.



instances with ﬁxed end points, which are used for connecting the                          instance is much larger, this approach can be quite time
clusters.                                                                                  consuming and can be applied only by a limited number of
   The divide-and-conquer approach optimizes the subsolutions                              iterations. Localization of the operations is therefore necessary in
and the overall tour independently. It is likely to lead to some                           this ﬁne-tuning step.
suboptimal choices, which makes it also possible to improve the
overall tour later by applying additional operators as a ﬁne-tuning                        Solutions From the Literature
step. This outcome can signiﬁcantly improve the solution,                                  In the literature, Karp (1977) applied a divide-and-conquer
especially across the cluster boundaries. Here, the same single-                           technique with a k-d tree-like structure in 1997. Two heuristics
solution TSP solver can be applied, but since the entire problem                           (including Lin and Kernighan) and one optimal heuristic



Frontiers in Robotics and AI | www.frontiersin.org                                     6                                            October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                                                   Santa Claus Challenge 2020




     FIGURE 6 | A straightforward approach (A) and the pseudocode approach (B) to generate open-loop, ﬁxed-start, and k-TSP (k  2) solutions from the closed-loop
     solution.



(dynamic programming) were considered for solving the                                     consisted of 1.9 M city locations. In 2020, Geonames6 was
subproblems. Problem sizes of N  128 were used. No TSP                                   published with 12 M geographical features on Earth. Also in
experiments were provided, but gap values of the                                          2020, GalaxyTSP was published (Drori et al., 2020) and extended
corresponding MST solutions using 16 clusters varied from                                 the largest size ever considered by containing 1.69 billion stars.
3.7 to 10.7%.                                                                                A quadtree-based divide-and-conquer technique was applied
   Valenzuela and Jones (1995) extended Karp’s idea by selecting                          by Drori et al. (2020) using LKH to solve the subproblems in
the direction of a cut via a genetic algorithm and by introducing a                       parallel. It took 50 min to solve the WorldTSP, reaching a 1% gap
method to merge the subsolutions. The results within a 1% gap to                          compared to the best known result. The current record holder is
the optimal were reported at the cost of 12–28 h of processing                            Keld Helsgaun (15th Feb. 2021), who used essentially the same
time for a problem instance of size N  5,000. Parallel                                   LKH method that was submitted here to the Santa competition.
implementation of Karp’s algorithm was described by Cesari                                To solve the GalaryTSP, the data were divided into 1,963 tiles
(1996), and processing times less than 1 h were reported using                            with an average size of 861,913 nodes, and it took approximately
16 processors for problem sizes up to N  7,937. Modest gap                               3 months to solve the problem in parallel (Drori et al., 2020).
values (approximately 5%) were reported with only two                                        Overall, the state of the art in the literature still appears to rely
subproblems.                                                                              mainly on local search and LKH. A variant of the LKH was
   Things started to develop signiﬁcantly in 2003 when Mulder                             compared against a genetic algorithm with an edge assembly
and Wunsch (2003) applied the adaptive resonance theory                                   crossover by Paul et al. (2019) to determine which method ﬁnds
algorithm of Carpenter and Grossberg (1998) to cluster data                               the optimal solution faster given a 1-h time limit. The genetic
by a hierarchical k-means variant in combination with a variation                         algorithm was better in 73% of the cases, but only problem sizes
of the Lin–Kernighan heuristic. It reached an 11% gap with a                              up to N  5,000 were considered. However, optimality was required,
problem size of N  1 M in which there were randomly                                      and the authors expected the processing times to align when the
distributed cities, and the solution spent only 16% of the time                           problem size increased. We cannot therefore conclude much about
required by the full version (24 min vs. 2 h 30 min). In the same                         how the genetic algorithm would perform with the Santa
year, 2003, the WorldTSP5 data set was published, which                                   competition, in which the success is measured by the gap value.


5                                                                                         6
    http://www.math.uwaterloo.ca/tsp/world                                                    https://www.geonames.org



Frontiers in Robotics and AI | www.frontiersin.org                                    7                                          October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                                            Santa Claus Challenge 2020




  FIGURE 7 | Comparison of Delaunay links and those used in the LKH submission.




Other Cases                                                                       only four algorithms were eventually submitted to the
A straightforward approach to support several TSP variants is to                  competition:
create a base algorithm that solves the closed-loop TSP and then
to make small modiﬁcations to the result to generate solutions for                      • Keld Helsgaun,
the open-loop, ﬁxed-start, and k-TSP cases. For example, an                             • Tilo Strutz,
open-loop solution (TSP path) can be created by simply cutting                          • UEF,
the longest link from the closed-loop variant (TSP tour). A ﬁxed                        • Anonymous.
start point can also be created in the same manner by cutting one
of the links connected to Santa’s home. The same idea generalizes                    Three worked as expected. They all were based on local search.
to the k-TSP case by removing k links (see Figure 6). However,                    The fourth failed to work even if it ran signiﬁcantly longer than
this variant is slightly naïve. For example, removing the longest                 the time limit. Originally, we did not plan to submit our own
link from the optimal closed-loop solution does not necessarily                   method because we were the organizers of the event. However,
create an optimal open-loop solution.                                             since the number of submissions remained small, we decided to
    A more sophisticated approach is to create a pseudonode that                  contribute ourselves by using our simple baseline solver from
has a constant distance to all other nodes (Sengupta et al., 2019).               Sengupta et al. (2019) with the clustering-based divide-and-
Because of the equal distances, it does not matter which nodes it                 conquer approach that we had implemented earlier to serve as
will be connected to. However, it allows the remainder of the tour                a reference solution. We did not expect it to perform well in this
to be optimized as a path and to avoid one long link somewhere                    competition.
along the tour. The two nodes connected to the pseudonode will
represent the start and end points of the open-loop variant. The                  Keld Helsgaun
chosen distance to the pseudonode does not matter if the TSP                      Keld Helsgaun submitted his algorithm known as LKH7 with
solver is optimal. However, it was noted by Fränti et al. (2021)                  minor modiﬁcations and with speciﬁc parameter settings. It is an
that it is better to use some large constant, whereas using zero                  efﬁcient implementation of the LKH that supports up to k  5.
values works poorly with some heuristics. A ﬁxed start point can                  The algorithm uses a best-improvement strategy, because it tries
be dealt with by setting its distance to the pseudonode to 0. This                all combinations and selects the best combination when updating
approach will avoid one inﬁnite link and guarantees that the                      the tours. This process is usually slow, but the author modiﬁed it
selected node will become one of the end nodes in the TSP path.                   to have much better efﬁciency by precomputing a candidate set of
                                                                                  links to be considered instead of trying all possible links. This
                                                                                  strategy is accomplished by building a graph that preserves the
SUBMITTED METHODS                                                                 neighborhood, such as using Delaunay triangulation (Reinelt,

In total, 82 trial TSP solutions were submitted to the testing
website by 13 different authors to see how they ranked. However,                  7
                                                                                      http://akira.ruc.dk/∼keld/research/LKH



Frontiers in Robotics and AI | www.frontiersin.org                           8                                             October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                                              Santa Claus Challenge 2020




  FIGURE 8 | An example of how alpha nearness is calculated for a given link with respect to the minimum 1-tree and the 1-tree that includes the forced link.




  FIGURE 9 | Greedy algorithm used by LKH and the result on the Santa data.



1992; Peterson, 1998). LKH supports several options, of which                         shown to eventually converge to the true alpha values, but
Keld Helsgaun considered a hybrid of Delaunay and Quadrant                            stopping it early is the key to high efﬁciency. This goal is
links when building the graph (see Figure 7). The quadrant graph                      accomplished in LKH by setting a small initial period for the
is formed using the nearest neighbors in each of the four                             ascent (INITIAL_PERIOD  100, default is N/2).
quadrants centered at a given point.                                                     A third necessary parameter setting is TIME_LIMIT. It is set
    The resulting links are prioritized using alpha-nearness: the                     to 3,000 s (50 min) because time starts ticking only after the
increase in cost when a minimum 1-tree is required to contain                         preprocessing is done: generating the candidate sets and
the link to be scored (see Figure 8), where a 1-tree is a spanning tree               estimating the alpha-nearness values take approximately 10 min.
deﬁned on a node set minus one special node combined with two                            The other parameters were the following:
links from that special node. A 1-tree is actually not a tree because it                 INITIAL_TOUR_ALGORITHM  GREEDY
contains a cycle. LKH is primarily designed for the closed-loop                          MAX_SWAPS  1,000 maximum number of swaps allowed in
variant, where the 1-tree concept is especially meaningful; a TSP tour                any search for a tour improvement
is a 1-tree where all nodes have a branching factor of 2. The optimum                    The greedy method used in LKH is explained in Figure 9.
TSP tour is the minimum 1-tree with all branching factors equal to 2.                 Despite the fact that greedy heuristics such as nearest insertion
    Computing all alpha-nearness values is time consuming O(N2)                       require quadratic time complexity and will not terminate during
and will complete in several days on an instance the size of Santa                    the course of 1 h on the Santa data, this version of Greedy can be
data. Fortunately, approximation is possible using subgradient                        computed efﬁciently due to the precalculated candidate set, which
optimization (Held and Karp, 1971). This process has been                             signiﬁcantly limits the choices at each step.



Frontiers in Robotics and AI | www.frontiersin.org                                9                                         October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                                       Santa Claus Challenge 2020




  FIGURE 10 | Grid-based division demonstrated (A) and the centroids after repartitioning (B).




  FIGURE 11 | Intermediate steps of DoLoWire visualized.




    As such, LKH was applied to the closed loop variant of TSP                        Tilo Strutz
and produced a winning tour length of 109,284 km. The other                           Tilo Strutz’s implementation is called DoLoWire, and its detailed
variants are supported with only minor modiﬁcations:                                  description is documented in his published work (Strutz, 2021). It
    A. For the open-loop case, a pseudonode was added. This                           ﬁrst clusters the points using a grid-based method. Because the
node was marked as a special node to be recognized by LKH, and                        data are not uniformly distributed, the grid adds more cells in
all distances from it to all others were considered to be equal                       regions with higher point density in an attempt to make the size of
to zero.                                                                              the subproblems smaller to be solvable in time. To control the
    B. For the ﬁxed-start case, the same pseudonode was added as                      number of clusters, DoLoWire has a parameter setting for the
in (A), but it was forced to be linked to Santa’s home.                               maximum value. This parameter was set to 1,500, and a total of
    C. For the k-TSP case, multiple pseudonodes were added to the                     1,268 clusters were generated on the Santa data (see Figure 10).
instance and considered to be depots in the multi-TSP variant of LKH.                                         √value performed better in practice than
                                                                                      Tilo Strutz noted that this
    Analyses of the generated solutions can be found in the                           at the theoretical best, N  1, 199. Once the cells are generated,
Evaluation section.                                                                   a k-means–like step is applied, where centroids are computed



Frontiers in Robotics and AI | www.frontiersin.org                               10                                   October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                                                Santa Claus Challenge 2020




  FIGURE 12 | Two-level clusters produced by k-means applied ﬁrst with k1  112 and then repeated for each cluster with k2  114 on each of the 112 clusters. A
  distribution of the cluster sizes is also shown. TSPDiv intermediate steps.




based on the data in each cell. Repartitioning is then performed                       updated immediately. One important aspect of the algorithm
according to the updated centroid locations. This approach                             design is that the maximum number of iterations of each
models the data better, because it avoids making an overly                             cluster is set to 5. This setting was used both when optimizing
arbitrary division imposed by the grid (see Figure 10).                                the coarse tour and at the cluster level. This value controls the
   The cluster centroids are used to generate a coarse tour                            processing time of the algorithm in such a way that a lower
(Figure 11). This step has a strong impact on the quality of                           value decreases the time at the cost of missing some
the ﬁnal solution, because it decides the general order in which                       improvements.
the points will be traversed. For this optimization, DoLoWire                             With the course tour computed, the next step is to ﬁnd suitable
uses a local search composed of 2- to 3-opt operations with the                        end points between consecutive cluster pairs. This step is
ﬁrst improvement strategy: when a better tour is found, it is                          accomplished using a simple search for the nearest pair across


Frontiers in Robotics and AI | www.frontiersin.org                                11                                          October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                            Santa Claus Challenge 2020




  FIGURE 13 | Proﬁling of submitted methods over the course of 1 h.



the cluster. This search requires quadratic time complexity.                   K-means takes O(Nk) and becomes slower with the increasing
However, since the number of clusters is a thousand times                  number of clusters (k). We therefore apply two-level
                                                                                                                             √     clustering
smaller than the complete data, this part is not a bottleneck.             with which we ﬁrst cluster the data into k1  3 N  112 groups,
Figure 11 shows the clusters with the chosen links.                        which are then clustered further by a second round of k-means
    Once the clusters and their connections have been decided, the         √3  makes the ﬁnal number of clusters proportional to k2 
                                                                           that
algorithm optimizes each cluster individually. It uses the same 2-            N2 ≈ 12, 764 (see Figure 12). The ﬁrst step is followed by
to 3-opt–based local search using the ﬁxed start and end points.           ﬁnding the end points and solving the TSP within each of these 112
The optimization can stop at a prior iteration when 2- and 3-              clusters in Level 2. A ﬁne-tuning step is applied where consecutive
optimality is already achieved. This process takes approximately           clusters are grouped together and optimized jointly. The result is a
40 min (see Figure 11, right).                                             coarse tour that passes through all 12,764 clusters. This process is
    The remaining 20 min are used for ﬁne-tuning. This step is             repeated to generate a complete tour (see Figure 12).
accomplished by splitting the tour into 500 long nonoverlapping                To complete in 1 h, we use the following parameter settings for
segments and optimizing the tour within each of them. This step            the random mix local search: ﬁve repeats each with N × 3,000
is applied in parallel and receives a strong boost here due to our         local search iterations at all levels; 1 repeat each with N × 3,000
hardware system being composed of 48 cores. This method is the             iterations at the ﬁne-tuning stage. The initial tour in the
only submission to use parallelization in some way. A ﬁnal closed          beginning is random. The ﬁnal tour length produced in this
loop tour of 115,620 km is reached.                                        way is 124,162 km before ﬁne-tuning and 122,226 km after ﬁne-
    To obtain the other variants, the closed-loop tour was                 tuning.
postprocessed as follows:                                                      We obtain the other variants as follows:

   – Open Loop: Cut the longest link;                                         – Open Loop: Do not close the coarse tour.
   – Fixed start: Cut the longest link connected to Santa’s home;             – Fixed start: Force Santa’s home and the corresponding
   – k-TSP: Cut the longest k links.                                            clusters to be the ﬁrst.
                                                                              – K-TSP: Cluster the data into k groups by k-means and solve
  Analysis of the generated solutions can be found in the                       them independently.
Evaluation section.
                                                                              We note that in the case of large-scale instances, unlike as
UEF                                                                        reported in Sengupta et al. (2019), the Link swap operator does
Our method turns out to be very similar to Tilo Strutz’s                   not play a signiﬁcant role because it operates with the end points,
submission, and below, we merely discuss the differences in                which have only minor contributions to the very long TSP path. It
their design. We use k-means instead of grid-based clustering              also does not apply at Level 2 within the clusters because their
and ﬁnd a much larger number of clusters (12,764 > 1,268). The             start and end points are ﬁxed when connecting the clusters. It also
reason is that our TSP solver, which uses random mix local search          does not apply to the closed-loop case. For these reasons, it has
(Sengupta et al., 2019), was designed for an order of magnitude            been disabled elsewhere except at Level 1 in the open-loop and in
smaller instances than what would be obtained by dividing the              the k-TSP cases. The components and the processing time proﬁles
data into √N clusters. Speciﬁcally, it can handle at most a few            of all of the submitted methods are summarized in Figure 13 and
hundred nodes in a reasonable time.                                        Table 1.




Frontiers in Robotics and AI | www.frontiersin.org                    12                                   October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                                                       Santa Claus Challenge 2020



TABLE 1 | Summary of the submitted methods.

                    Keld Helsgaun                                                  Tilo Strutz                          UEF

Method:             LKH                                                            DoLoWire                             TSPDiv
Localization        Neighbors                                                      Grid                                 Clusters
Local search        Best improvement: 2- to 5-opt with alpha-prioritization        First improvement: 2- to 3-opt       First improvement: 2-opt, node swap and Link swap
Dividing            —                                                              Grid-based clustering                Two-level k-means
Merging             —                                                              Two nearest                          Two nearest
Fine-tuning         —                                                              Fixed-length segments (500)          Consecutive cluster pairs
Other variants      Pseudonode                                                     Postprocessing                       Pseudonode




  FIGURE 14 | Summary of the results of the Santa competition. The difference between the 1st (109,284 km) and the 3rd (122,226 km) is approximately 12%.
  When repeated 10 times, the standard deviations of the results were only 98 km (LKH), 47 km (DoLoWire) and 143 km (TSP Div), which shows that the effect of
  randomness is negligible, which originates from the large number of trial operators applied and the large problem size. According to ANOVA, the results were statistically
  signiﬁcant (p < 1030 in all cases).




EVALUATION                                                                                 use any divide-and-conquer technique to reach the 1-h time limit.
                                                                                           A detailed example of the optimized tours is shown in Figure 15.
The results of the submitted methods are summarized in                                        The results for the open-loop and ﬁxed start point variants
Figure 14. The main observation is that the performance                                    have the same ranking. The results of the k-TSP case show the
difference between the methods is clear. The tour length of                                importance of a proper task deﬁnition. The purpose was to use
LKH is 109,284 km, and it is clearly the shortest of all. The gap                          Santa’s helpers to deliver all of the goodies as fast as possible, but
to the second best (DoLoWire) is approximately 6% and to the                               the exact objective was not deﬁned. As a result, LKH and
third best (UEF) is 12%. What is surprising is that LKH does not                           DoLoWire minimize merely the total length of all of the



Frontiers in Robotics and AI | www.frontiersin.org                                    13                                            October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                                                 Santa Claus Challenge 2020




  FIGURE 15 | Tours of each of the submitted methods zoomed into view in the city of Joensuu, downtown area (above), and the city area including suburban areas.




  FIGURE 16 | Different graphs and the corresponding number of links. Most people in Finland live in the South, where data are denser than those in the North.



subtours, while a more appropriate objective function would have                        Peterson (1998) that 75% of the links in an optimal TSP path also
been to minimize the length of the longest subtour. LKH provides                        appear in the MST and 97% in the XNN graph, which is a
four subtours with lengths that vary from 1 to 85,318 km, which is                      subgraph of the much larger Delaunay graph used in LKH. Thus,
not meaningful for Santa’s task. The UEF submission was the                             the chosen neighborhood graph is likely to not miss much.
only submission that provides a somewhat balanced workload                                  There are also other nearest neighborhood graphs, such as
with subtour lengths that vary from 15,203 to 42,936 km.                                KNN, XNN, ε-epsilon neighborhood, Delaunay, Gabriel, MST,
                                                                                        and k-MST. Some of them require parameters such as the number
Component-Level Analyses (Closed Loop)                                                  of nearest neighbors (KNN), rounds of the algorithm (k-MST), or
Localization                                                                            size of the neighborhood (ε-neighborhood). Setting their values
LKH was powerful enough to provide the winning tour by solving                          larger would increase the number of connections, which would
the problem as a whole without division into subproblems. This                          reduce the probability of missing good links but would also
ﬁnding implies that the use of a neighborhood graph is important.                       increase the search space and slow down the search. This
It could miss some links that belong to the optimal tour, but                           methodology needs to be balanced somehow. Another issue
according to the results, its effect is still much less than the effect                 is the connectivity. Other graphs (XNN, Delaunay, Gabriel, and
of the divide-and-merge strategy. It was reported in the work by                        MST) guarantee connectivity and do not require any


Frontiers in Robotics and AI | www.frontiersin.org                                 14                                         October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                                              Santa Claus Challenge 2020




  FIGURE 17 | Results when applying 3 different local search strategies when optimizing the same set of clusters and merging along the same coarse tour. The
  between-cluster connections are shown in red.



parameters. Their sizes vary in such a way that MST ⊂ Gabriel                         Local Search
 ⊂ Delaunay; Delaunay provides the largest number of links                            We compare the three search strategies:
(see Figure 16).
   Keld Helsgaun considered a hybrid of Delaunay and quadrant                            – 2- to 5-opt with neighbor candidates and alpha prioritization
links when building the graph. The quadrant graph is formed using                          (Keld Helsgaun)
nearest neighbors in each of the four quadrants centered at a given                      – 2- to 3-opt (Tilo Strutz)
point. This method was previously shown to work well in clustered                        – 2-opt, Relocate, and Link swap (UEF)
data (Helsgaun, 2000), which is the case with geographical
instances such as ours, where cities form clear, dense clusters.                         We ﬁxed the division method and the grid-based clustering of
Only a maximum of ﬁve Delaunay links and four quadrant links                          Tilo Strutz. We also ﬁxed the order of the coarse tour, as shown
per node were considered. The question of how to choose these                         in Figure 17, in such a way that the resulting solutions would
values for the best performance remains an open problem.                              differ only inside the clusters. We attempted to solve the same
   We experimented with LKH with the complete Delaunay                                clusters using all three methods shown above over the course of
graph instead of the hybrid combination while keeping all of                          40 min to allow for 20 min of ﬁne-tuning (see the Clustering
the other parameters the same. This approach further improved                         section). To complete the computations in approximately
the tour by 0.3% to 108,996 km with the same processing time.                         40 min, we set LKH to terminate after at the most 3 s per
This ﬁnding shows that the choice of the neighborhood graph still                     cluster and set the number of random mix repeats to 1, and
has room for improvement. Although the data show clear clusters                       the iterations equal to 11,000 × the size of the cluster. For
indicating that the hybrid combination of links should work well,                     DoLoWire, we used the default settings. All of these settings
other characteristics also have an impact. Finland shows                              terminated in approximately 40 min.
signiﬁcant differences in density from south to north.                                   From the results in Figure 17, we note that the solution
Approximately 20% of the population lives in the Helsinki                             length varies signiﬁcantly in the three cases. Even though they
metropolitan area and 50% in south Finland8 (see Figure 16).                          do not look much different in the zoomed out view, zooming in
Finland also has thousands of lakes that have a signiﬁcant impact                     reveals the limitations of random mix in solving such large
on optimal routing, especially in East Finland.                                       cluster sizes properly. Many suboptimal choices are visible. It is
   Clustering allows solving larger-scale instances more                              good on small instances but never converges with larger
efﬁciently, but it also has a role in localizing the search space                     instances because of a search space that is too large. The 2-
by limiting the search within the cluster. Another beneﬁt is that it                  to 3-opt provides better results, and substantially fewer artifacts
supports direct parallelization, where one cluster could be solved                    are visible. The 2- to 5-opt with alpha-prioritization produces
by one processor if enough CPU resources are available.                               the best result despite the extra time required to compute the
                                                                                      Delaunay graph for each cluster. The more powerful operator
                                                                                      and the selection strategy compensate for the extra time. The
8
  https://www.verkkouutiset.ﬁ/karttakuva-nayttaa-puolet-suomalaisista-asuu-           result (113,598 km) is better than any of the submitted divide-
taman-viivan-alapuolella                                                              and-merge variants.



Frontiers in Robotics and AI | www.frontiersin.org                               15                                        October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                                            Santa Claus Challenge 2020




  FIGURE 18 | The data clustered by Strutz’ grid-based method (A) and k-means (B). The same number of clusters was used in both (k  1,268). Example tours
  through the centroids are shown. Histograms of the cluster sizes are also shown.



Clustering                                                                           lost when a single centroid is used to represent the cluster. The
We next compare the two clustering methods (grid and k-means)                        coarse tour is forced to move west from here, because going back
in Figure 18. Strutz’s grid-based method with his parameter                          to the south is not possible (returning to the same cluster). The
choices produced 1,268 clusters, and thus, we also set the same                      grid-based clusters are smaller here and allow the coarse tour to
value for k-means to have a fair comparison.                                         return to the south. The grid cells are small enough to better
    The centroids in the grid clustering are more uniformly                          preserve these patterns. The merged solution in k-means also has
distributed over the country, and north Finland is better covered                    visible artifacts where the solution could be easily improved (see
by the clusters than using k-means. This ﬁnding occurs because grid-                 Figure 20).
based clustering has a ﬁxed size of the cells independent of the                         The south is modeled better by k-means, but the improvement
volume. K-means clusters vary more: larger clusters appear in the                    is less signiﬁcant. The reason is that in high-density areas, there
north, and smaller clusters appear in the south. The coarse tour for                 are plenty of alternative choices, and it is easier to ﬁnd many
the grid clusters has links of roughly equal sizes (distances between                almost equally good alternatives. In the low-density areas in the
the cells) compared to the tour for k-means.                                         north, even a single mistake can have a serious impact on the
    We used LKH to solve the individual clusters. In preparation                     overall tour length, because there are fewer alternatives and
for the merging step to follow, we ﬁxed the start and end points                     unnecessary detours can mean tours that are hundreds of
for each cluster to be the nearest points between the consecutive                    kilometers longer. In other words, sparsely populated areas are
clusters in the coarse tour. We can impose the ﬁxed start and end                    more critical for performance than high-density areas.
points to be used by LKH using the pseudonode strategy and two                           We also experimented by changing the parameters that
ﬁxed links.                                                                          control the cell size, but the value of k  1,268 appears to be
    The grid clustering produces a better result. The main reason                    locally optimal, and it is difﬁcult to ﬁnd signiﬁcant improvement.
is that it better represents the northern part of Finland. K-means                   There are simply too many possible parameter combinations to
generates clusters with large volumes but loses information about                    test, and the high computing time is a serious restriction to
how those points are distributed (see Figure 19). For example, we                    running large-scale √testing.
                                                                                                                 The selected value is quite close to the
note that the north-most convex hull contains a visible looping                      theoretical best N  1, 199, and thus, we do not expect
path formed by buildings along main roads; this information is                       signiﬁcant improvement here.



Frontiers in Robotics and AI | www.frontiersin.org                              16                                        October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                                                       Santa Claus Challenge 2020




  FIGURE 19 | Solutions after solving individual clusters and merging the subtours. The convex hulls of the clusters are given for reference. On the right, data points
  are also shown to highlight the patterns.




  FIGURE 20 | Result of ﬁne-tuning using two techniques and an example where optimization is not possible using either of the two techniques.



   In conclusion, grid clustering is better for this purpose because it                     Fine Tuning
better models the low-density areas where mistakes have a more severe                       We tested the two ﬁne-tuning strategies by applying the LKH
impact on the overall tour length. In the following tests, we ﬁx LKH to                     solver for subsets produced as follows:
use alpha prioritization on a full Delaunay graph to avoid the need for
parameter tuning and because it was shown to produce better results                            – Nonoverlapping segments of length 500.
than the original heuristic used by LKH (5 Delaunay, 4 Quad hybrid).                           – Adjacent cluster pairs.



Frontiers in Robotics and AI | www.frontiersin.org                                     17                                           October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                                              Santa Claus Challenge 2020



TABLE 2 | Summary of the results. The baselines of the three submitted methods are in boldface.

                                     Operator            Localization               Divide                 Clusters             Fine-tuning             Result (km)

Greedy                                   —               —                          —                         —                 —                         129,775
UEF and DoLoWire:
  UEF (no tuning)                    Rand mix            Clusters                   K-means                 12,764              —                         124,162
  UEF                                Rand mix            Clusters                   K-means                 12,764              Two clusters              122,226
  UEF (grid clusters)                Rand mix            Clusters                   Grid                     1,268              Two clusters              147,817
  DoLoWire (no tuning)              2- to 3-opt          Grid                       Grid                     1,268              —                         120,751
  DoLoWire                          2- to 3-opt          Grid                       Grid                     1,268              500 points                115,620
LKH with clustering:
  LKH (k-means)                     2- to 5-opt          Neighbors                  K-means                 12,764              —                         121,020
  LKH (k-means)                     2- to 5-opt          Neighbors                  K-means                  1,268              —                         117,496
  LKH (grid)                        2- to 5-opt          Neighbors                  Grid                     1,268              —                         113,598
  LKH (grid + tuning)               2- to 5-opt          Neighbors                  Grid                     1,268              500 points                112,404
  LKH (grid + tuning)               2- to 5-opt          Neighbors                  Grid                     1,268              Two clusters              111,636
LKH without clustering:
  LKH                               2- to 5-opt          Neighbors                  —                         —                 —                         109,284
  LKH (Delaunay)                    2- to 5-opt          Delaunay                   —                         —                 —                         108,996




   We applied ﬁne-tuning for the Grid+LKH combination shown                         because of quadratic time complexity. To solve problems of this size
in Figure 19 (left). We set a maximum processing time of 2 s per                    in 1 h, a linear (or close to linear) algorithm is required.
subset. Each time, the initial tour was simply the combined                            We also draw the following conclusions:
subtours of the two segments (or clusters). Both ﬁne-tuning
strategies terminated in approximately 20 min each. The                                   • Spatial localization of the local search operator is most
results are shown in Figure 20. The adjacent clusters strategy                              important.
is better. However, both strategies optimize merely among the                             • Local search with k-opt is still the state of the art.
pairs of points that are near to each other in the tour sequence but                      • The k-opt needs to be extended to 4-opt and 5-opt.
not necessarily in the space. We expect a better ﬁne-tuner to be                          • Current divide-and-merge strategies requires further
constructed by selecting the subsets from sequences (or clusters)                           improvements.
nearby in the space but far away in the tour sequence.                                    • Parallelization would be an easy addition to speed-up the
                                                                                            methods further.
Summary of the Results
The main results are summarized in Table 2. The ﬁrst observation is                     In speciﬁc, without the neighborhood graph, the k-opt and
that localization is more effective than dividing. LKH (109,284 km)                 random mix operators produced three orders of magnitude worse
achieves the best results even if ﬁne-tuning was applied after                      solutions because of the huge search space. Random initialization was
clustering (111,636 km). The role of clustering therefore remains                   another limitation, but thanks to the localization by neighborhood
as facilitating parallel processing. The second observation is that the             graph, greedy initialization could be calculated efﬁciently.
localization can be improved using Delaunay alone instead of                            While there were only three valid submissions, they were all
combining it with the quadrant neighbors as in the submission.                      based on local search. Literature review did not reveal any other
The third observation is that the effect of ﬁne-tuning is most                      method than local search capable of scaling up to data of the size
signiﬁcant with DoLoWire (120,751 vs. 115,620 km) because of                        of 1 M. In one article, GA was found to be more effective than
the effect of parallelization. Without parallelization, UEF ﬁne-                    local search but only up to N  5,000, and when needed to ﬁnd the
tuning method (adjacent clusters) is better. Among the other                        exact optimum (Paul et al., 2019). Probably the strongest evidence
parameters, the number of clusters is also remarkable. Too many                     of local search being the state of the art is that the method by Keld
clusters are bad with LKH (121,020 vs. 117,496 km), whereas too few                 Helsgaun has held the record for the other large-scale benchmark
clusters are bad for Rand mix (122,226 vs. 147,817 km) because the                  data set (WorldTSP), almost without a break since 2003.
clusters are large, and the local operators lose their effectiveness.                   About the chosen data from OSM: it was ﬁt for the purpose but
                                                                                    suffered some artifacts because the coverage of the data varied a
                                                                                    lot. While the data is valid, a more accurate building distribution
CONCLUSION                                                                          in Finland would be available.9
                                                                                        We derived the best divide-and-conquer approach from
We studied three solutions for a large-scale TSP problem in the Santa               the components of the submitted variants and reached a solution
Claus challenge in 2020. From the results, we learned a few important               with a 2% gap to the best method (LKH). This ﬁnding is signiﬁcantly
lessons. First, large-scale instances have immediate consequences that              better than the gap values of the two other submitted methods
must be taken into account when designing algorithms for big data.
Size of 1.4 M is already so large that even a simple greedy algorithm
would take about 3 days with our current hardware to compute                        9
                                                                                        https://www.avoindata.ﬁ/data/ﬁ/dataset/rakennusten-osoitetiedot-koko-suomi



Frontiers in Robotics and AI | www.frontiersin.org                             18                                           October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                                                           Santa Claus Challenge 2020



(DoLoWire  6%, UEF  12%). The potential of the divide-and-                                 AUTHOR CONTRIBUTIONS
conquer approach comes from the fact that the calculations could be
easily performed in parallel using a multiprocessor system. This                             The authors PF and RM-I organized the competition and wrote
ﬁnding also applies to the ﬁne-tuning step of the 1,267 merged cluster                       the article jointly. The author RM-I performed all the
pairs. With a state-of-the-art cloud infrastructure, we could run                            experiments and implemented the TSP Santa Claus challenge
one task per machine (Mariescu-Istodor et al., 2021), which                                  webpage.
would bring the processing time from 1 h down to 2 min.

                                                                                             SUPPLEMENTARY MATERIAL
DATA AVAILABILITY STATEMENT
                                                                                             The Supplementary Material for this article can be found online at:
All data is available in the Santa Claus TSP challenge web page:                             https://www.frontiersin.org/articles/10.3389/frobt.2021.689908/
https://cs.uef.ﬁ/sipu/santa/.                                                                full#supplementary-material


                                                                                             Golden, Bruce. L., Raghavan, Subramanian., and Wasil, Edward. A. (2008). The
REFERENCES                                                                                       Vehicle Routing Problem: Latest Advances and New Challenges. New York:
                                                                                                 Springer Science & Business Media.
Applegate, David., Bixby, Robert. E., Chvátal, Vasek., and William, J. C. (1999).            Held, Michael., and Karp, Richard. M. (1971). The Traveling-Salesman Problem
   Concorde: a Code for Solving Traveling Salesman Problems. Available at:                       and Minimum Spanning Trees: Part II. Math. Programming 1, 16–25.
   http://www.tsp.gatech.edu/concorde.html.                                                      doi:10.1007/bf01584070
Applegate, David. L., Bixby, Robert. E., Chvátal, Vašek., and William, J. (2011). The        Helsgaun, K. (2000). An Effective Implementation of the Lin-Kernighan Traveling
   Traveling Salesman Problem. Princeton University Press.                                       Salesman Heuristic. Eur. J. Oper. Res. 126 (1), 106–130. doi:10.1016/s0377-
Arnold, F., Gendreau, M., and Sörensen, K. (2019). Efﬁciently Solving Very Large-                2217(99)00284-2
   Scale Routing Problems. Comput. Operations Res. 107, 32–42. doi:10.1016/                  Jiang, J., Gao, J., Li, G., Wu, C., and Pei, Z. (2014). “Hierarchical Solving Method for
   j.cor.2019.03.006                                                                             Large Scale TSP Problems,” in International symposium on neural networks
Barnhart, C., Johnson, E. L., Nemhauser, G. L., Savelsbergh, M. W. P., and Vance,                (Cham: Springer), 252–261. doi:10.1007/978-3-319-12436-0_28
   P. H. (1998). Branch-and-price: Column Generation for Solving Huge Integer                Karp, R. M. (1977). Probabilistic Analysis of Partitioning Algorithms for the
   Programs. Operations Res. 46 (3), 316–329. doi:10.1287/opre.46.3.316                          Traveling-Salesman Problem in the Plane. Mathematics OR 2 (3), 209–224.
Bixby, Robert. (2007). The Gurobi Optimizer. Transportation Res. B 41 (2),                       doi:10.1287/moor.2.3.209
   159–178. doi:10.1016/s0965-8564(07)00058-4                                                Kobayashi, K. (1998). “Introducing a Clustering Technique into Recurrent Neural
Braekers, K., Ramaekers, K. I., and Van Nieuwenhuyse, Inneke. (2016). The                        Networks for Solving Large-Scale Traveling Salesman Problems,” in
   Vehicle Routing Problem: State of the Art Classiﬁcation and Review.                           International Conference on Artiﬁcial Neural Networks (London: Springer),
   Comput. Ind. Eng. 99, 300–313. doi:10.1016/j.cie.2015.12.007                                  935–940. doi:10.1007/978-1-4471-1599-1_146
Carpenter, Gail. A., and Grossberg, Stephen. (1998). The ART of Adaptive                     Krumm, John., and Horvitz, Eric. (2017). Risk-Aware Planning: Methods and Case
   Pattern Recognition by a Self-Organizing Neural Network. Computer 21 (3),                     Study for Safer Driving Routes. San Francisco: AAAI, 4708–4714.
   77–88.                                                                                    Kuo, Yiyo., and Wang, Chi-Chang. (2011). Optimizing the VRP by Minimizing
Cesari, G. (1996). Divide and Conquer Strategies for Parallel TSP Heuristics.                    Fuel Consumption. Manag. Environ. Qual. Int. J. 22 (4), 440–450. doi:10.1108/
   Comput. Operations Res. 23 (7), 681–694. doi:10.1016/0305-0548(95)00066-6                     14777831111136054
Chitty, D. M. (2017). “Applying ACO to Large Scale TSP Instances,” in UK                     Li, Xiaojiang., Yoshimura, Yuji., Tu, Wei., and Ratti, Carlo. (2019). A Pedestrian
   Workshop on Computational Intelligence (Cham: Springer), 104–118.                             Level Strategy to Minimize Outdoor Sunlight Exposure in Hot Summer. arXiv
   doi:10.1007/978-3-319-66939-7_9                                                               [Epub ahead of print].
Croes, G. A. (1958). A Method for Solving Traveling-Salesman Problems.                       Liao, E., and Liu, C. (2018). A Hierarchical Algorithm Based on Density Peaks
   Operations Res. 6 (6), 791–812. doi:10.1287/opre.6.6.791                                      Clustering and Ant colony Optimization for Traveling Salesman Problem. IEEE
Dantzig, G. B., and Ramser, J. H. (1959). The Truck Dispatching Problem. Manag.                  Access 6, 38921–38933. doi:10.1109/access.2018.2853129
   Sci. 6 (1), 80–91. doi:10.1287/mnsc.6.1.80                                                Lin, C., Choy, King. Lun., Ho, G. T. S., Chung, S. H., and Lam, H. Y. (2014). Survey
Drori, Iddo., Kates, Brandon., Sicklinger, William., Kharkar, Anant., Dietrich,                  of green Vehicle Routing Problem: Past and Future Trends. Expert Syst. Appl. 41
   Brenda., Shporer, Avi., et al. (2020). “GalaxyTSP: A New Billion-Node                         (4), 1118–1138. doi:10.1016/j.eswa.2013.07.107
   Benchmark for TSP,” in 1st Workshop on Learning Meets Combinatorial                       Mariescu-Istodor, Radu., Cristian, Alexandru., Negrea, Mihai., and Cao, Peiwei.
   Algorithms (Vancouver, Canada: NeurIPS).                                                      (2021). VRPDiv: A Divide and Conquer Framework for Large Vehicle Routing
Dumas, Y., Desrosiers, J., and Soumis, F. (1991). The Pickup and Delivery Problem                Problem Instances. Manuscript (submitted).
   with Time Windows. Eur. J. Oper. Res. 54 (1), 7–22. doi:10.1016/0377-2217(91)             Mulder, S. A., and Wunsch, D. C. (2003). Million City Traveling Salesman Problem
   90319-q                                                                                       Solution by divide and Conquer Clustering with Adaptive Resonance Neural
Fränti, Pasi., Nenonen, Teemu., and Yuan, Mingchuan. (2021). Converting MST to                   Networks. Neural Netw. 16 (5-6), 827–832. doi:10.1016/S0893-6080(03)00130-8
   TSP Path by branch Elimination. Appl. Sci. 11 (177), 1–17.                                Paul, Mc. Menemy., Veerappen, Nadarajen., Adair, Jason., and Ochoa, Gabriela.
Fränti, P., Mariescu-Istodor, R., and Sengupta, L. (2017). O-mopsi: Mobile                       (2019). “Rigorous Performance Analysis of State-Of-The-Art TSP Heuristic
   Orienteering Game for Sightseeing, Exercising, and Education. ACM                             Solvers,” in Theory and Applications of Models of Computation (Kitakyushu,
   Trans. Multimedia Comput. Commun. Appl. 13 (4), 1–25. doi:10.1145/                            Japan: Springer), 99–114.
   3115935                                                                                   Peterson, Samuel. (1998). Computing Constrained Delaunay Triangulations in
Fränti, P., Mariescu-Istodor, R., and Zhong, C. (2016). “XNN Graph,” in Joint Int.               the Plane. Available at: http://www.geom.uiuc.edu/∼samuelp/del_project.
   Workshop on Structural, Syntactic, and Statistical Pattern Recognition (S+SSPR)               html.
   (Merida, Mexico, LNCS 10029: springer international publishing), 207–217.                 Poikonen, S., Golden, B., and Wasil, E. A. (2019). A Branch-and-bound Approach
   doi:10.1007/978-3-319-49055-7_19                                                              to the Traveling Salesman Problem with a Drone. Informs J. Comput. 31 (2),
Gendreau, M., Hertz, A., and Laporte, G. (1992). New Insertion and                               335–346. doi:10.1287/ijoc.2018.0826
   Postoptimization Procedures for the Traveling Salesman Problem.                           Reinelt, Gerhard. (1992). A Traveling Salesman Problem Library. J. Oper. Res. Soc.
   Operations Res. 40 (6), 1086–1094. doi:10.1287/opre.40.6.1086                                 11 (1), 19–21.




Frontiers in Robotics and AI | www.frontiersin.org                                      19                                             October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---

Mariescu-Istodor and Fränti                                                                                                                        Santa Claus Challenge 2020



Sakurai, Yoshitaka., Onoyama, Takashi., Sen, Kubota., Nakamura, Yoshihiro., and             Xiang, Zuoyong., Chen, Zhenyu., Gao, Xingyu., Wang, Xinjun., Di, Fangchun., Li,
   Tsuruta, Setsuo. (2006). “A Multi-World Intelligent Genetic Algorithm to                    Lixin., et al. (2015). Solving Large-Scale Tsp Using a Fast Wedging Insertion
   Interactively Optimize Large-Scale TSP,” in IEEE International Conference                   Partitioning Approach. Math. Probl. Eng. 2015, 854218. doi:10.1155/2015/
   on Information Reuse & Integration, Waikoloa, HI, USA, 16-18 Sept. 2006                     854218
   (IEEE), 248–255. doi:10.1109/iri.2006.252421                                             Yang, Jin-Qiu., Yang, Jian-Gang., and Chen, Gen-Lang. (2009). “Solving Large-
Sengupta, L., Mariescu-Istodor, R., and Fränti, P. (2019). Which Local Search                  Scale TSP Using Adaptive Clustering Method,” in IEEE International
   Operator Works Best for the Open-Loop TSP? Appl. Sci. 9 (19), 3985.                         Symposium on Computational Intelligence and Design, Changsha, China,
   doi:10.3390/app9193985                                                                      12-14 Dec. 2009 (IEEE), 49–51. doi:10.1109/ISCID.2009.19
Shen, Lin., and Kernighan, Brian. W. (1973). An Effective Heuristic Algorithm for
   the Traveling-Salesman Problem. Operations Res. 21 (2), 498–516.                         Conﬂict of Interest: The authors declare that the research was conducted in the
Strutz, T. (2021). Travelling Santa Problem: Optimization of a Million-Households           absence of any commercial or ﬁnancial relationships that could be construed as a
   Tour within One Hour. Front. Robot. AI 8, 652417. doi:10.3389/                           potential conﬂict of interest.
   frobt.2021.652417
Taillard, É. D., and Helsgaun, K. (2019). POPMUSIC for the Travelling Salesman              Publisher’s Note: All claims expressed in this article are solely those of the authors
   Problem. Eur. J. Oper. Res. 272 (2), 420–429. doi:10.1016/j.ejor.2018.06.039             and do not necessarily represent those of their afﬁliated organizations, or those of
Thibaut, Lust., and Jaszkiewicz, Andrzej. (2010). Speed-up Techniques for Solving           the publisher, the editors, and the reviewers. Any product that may be evaluated in
   Large-Scale Biobjective TSP. Comput. Operations Res. 37 (3), 521–533.                    this article, or claim that may be made by its manufacturer, is not guaranteed or
Uchoa, E., Pecin, D., Pessoa, A., Poggi, M., Vidal, T., and Subramanian, A.                 endorsed by the publisher.
   (2017). New Benchmark Instances for the Capacitated Vehicle Routing
   Problem. Eur. J. Oper. Res. 257 (3), 845–858. doi:10.1016/j.ejor.2016.                   Copyright © 2021 Mariescu-Istodor and Fränti. This is an open-access article
   08.012                                                                                   distributed under the terms of the Creative Commons Attribution License (CC
Valenzuela, Christina. L., and Jones, Antonia. J. (1995). “A Parallel Implementation        BY). The use, distribution or reproduction in other forums is permitted, provided the
   of Evolutionary divide and Conquer for the TSP,” in First International                  original author(s) and the copyright owner(s) are credited and that the original
   Conference on Genetic Algorithms in Engineering Systems: Innovations and                 publication in this journal is cited, in accordance with accepted academic practice.
   Applications, Shefﬁeld, UK, 12-14 Sept. 1995 (IET), 499–504. doi:10.1049/cp:             No use, distribution or reproduction is permitted which does not comply with
   19951098                                                                                 these terms.




Frontiers in Robotics and AI | www.frontiersin.org                                     20                                            October 2021 | Volume 8 | Article 689908


--- PAGE BREAK ---


