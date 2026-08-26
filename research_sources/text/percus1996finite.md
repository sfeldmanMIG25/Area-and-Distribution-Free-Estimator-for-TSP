# Finite Size and Dimensional Dependence in the {Euclidean} Traveling Salesman Problem

> Citation key: `percus1996finite`  
> DOI: 10.1103/PhysRevLett.76.1188  
> Download URL: https://scholar.cgu.edu/allon-percus/wp-content/uploads/sites/11/2013/08/tspprl.pdf  
> SHA-256: `08af06263abb52f6f6f3a862646f4192db00be4aff324533db64863fc2b21d9c`  

---

VOLUME 76, NUMBER 8                     PHYSICAL REVIEW LETTERS                                           19 FEBRUARY 1996


       Finite Size and Dimensional Dependence in the Euclidean Traveling Salesman Problem
                                          Allon G. Percus* and Olivier C. Martin†
       Division de Physique Théorique, Institut de Physique Nucléaire, Université Paris-Sud, F-91406 Orsay Cedex, France
                                                     (Received 20 July 1995)
                We consider the Euclidean traveling salesman problem for N cities randomly distributed in the
             unit d-dimensional hypercube, and investigate the finite size scaling of the mean optimal tour
             length LE . With toroidal boundary conditions we find, motivated by a remarkable universality in
             the kth nearest neighbor distribution, that LE sd ­ 2d ­ s0.7120 6 0.0002d N 1y2 f1 1 Os1yNdg and
             LE sd ­ 3d ­ s0.6979 6 0.0002d N 2y3 f1 1 Os1yNdg. We then consider a mean-field approach in the
             limit N ! ` which we find to be a good approximation
                                                             p       (the error being less than 2.1% at d ­ 1, 2,
             and 3), and which suggests that LE sdd ­ N 121yd dy2pe spdd1y2d f1 1 Os1yddg at large d.

             PACS numbers: 02.60.Pn, 02.70.Lq, 64.60.Cn

   The traveling salesman problem (TSP) is one of the            function of dimension. Comparing mean-field results with
best known combinatorial optimization problems. It is            Euclidean N ! ` results at low d shows that mean field
NP complete (suggesting that no algorithm exists for             does considerably better than previously expected, and
solving the problem in polynomial time), and it serves as        suggests that in quite natural units, LE can be written as a
a fertile ground for analytical and numerical approaches to      power series in 1yd.
optimization problems in general. It is also one of the few         Euclidean model: Finite size scaling sd ­ 2d.—We
optimization problems that have been studied extensively         start with the case of N cities distributed randomly
in the context of statistical mechanics.                         and uniformly in a unit square. Numerous heuristic
   The TSP, as we consider it, is as follows: Given N            approaches have been developed to find near-optimal
points (“cities”) in a space, the problem is to find the         TSP tours given a particular configuration (“instance”) of
length of the shortest closed path (“tour”) going through        cities. For our purposes, the most convenient methods are
each city exactly once. Two particular forms of the prob-        local-optimization heuristics such as the Lin-Kernighan
lem have been investigated in depth. The first, which has        (LK) [4] and the chained local optimization (CLO) [5]
attracted the most attention among computer scientists and       algorithms. With these algorithms, repeated runs on a
mathematicians, is the Euclidean TSP: The N cities are           given instance using different random starts produce the
randomly distributed in a d-dimensional hypercube and            optimal tour with increasing probability.
the distances between cities are given by the Euclidean             It has been shown [6] that in the large N limit the
metric. The second, which has been of particular interest        optimal tour length for a given instance L̃E is self-
within the statistical physics community, is the random          averaging up to a scaling factor
link TSP: The lengths lij separating cities i and j are
                                                                                           L̃E
taken as independent random variables with a given dis-                              lim   121yd
                                                                                                   ­ bE ,
                                                                                     N!` N
tribution rsld.
   It has been noted by Mézard and Parisi [1] that the           where convergence to the instance-independent bE is with
random link model, with rsld appropriately chosen, maps          probability 1 (in the ensemble of instances with randomly
onto the Euclidean model if correlations between three or        distributed cities). Much past work has concentrated on
more distances are neglected (no triangle inequality, for        optimizing single instances at large N (see [5,7,8]). Here,
instance). This suggests that the random link TSP can be         however, our concern is to calculate bE along with an
considered as a mean-field approximation to the Euclidean        estimate of statistical error, and so instead we average
case, and perhaps that this approximation becomes exact          over a large number of instance. There is necessarily
in the limit d ! `.                                              a tradeoff in the choice of N: At small N alone we
   Our intention in this Letter is twofold. First, for           cannot confidently predict the finite size scaling behavior,
the Euclidean TSP we investigate finite size corrections         whereas at large N the large amount of computing time
to the mean optimal tour length LE , in the large N              necessary for each optimization sharply limits the number
(“thermodynamic”) limit. To our knowledge there has              of instances we can optimize reliably, and increases the
been no prior work on this subject, in spite of a great          statistical error. We therefore choose several small values
deal of interest in LE in the thermodynamic limit itself.        of N (N ­ 12 through N ­ 17) where we optimize using
Second, we explore the dimensional dependence of LE              LK, and two larger values (N ­ 30 and N ­ 100) where
using a mean-field approach (the random link TSP in              we optimize using CLO.
conjunction with the “cavity method” [1,2]). We extend              Given LE sNd at different values of N, then, we wish
the work of Krauth and Mézard [3] to find the mean-              to extrapolate and extract the limit bE , as well as finite
field optimum LMF in the thermodynamic limit, as a               size corrections. In order to eliminate the effects of

1188                 0031-9007y96y76(8)y1188(4)$06.00            © 1996 The American Physical Society


--- PAGE BREAK ---

VOLUME 76, NUMBER 8                    PHYSICAL REVIEW LETTERS                                          19 FEBRUARY 1996

surface terms, we use periodic boundary conditions in
the Euclidean distance metric. An indication of the size
dependence to be expected in LE sNd may be found
by looking at the distance Dk between kth nearest
neighbors, averaged over the ensemble of instances. A
direct calculations shows that, given N cities distributed
randomly and uniformly over the d-dimensional unit
hypercube (with periodic boundary conditions),
               µ        ∂          ∑              ∏k
                 N 21                    p dy2
 Dk sN, dd ,              sN 2 kdd
                 k21                 Gsdy2 1 1d
                  Z 1y2      ∑                 ∏N2k21
                          dk         p dy2
               3        r 12                          dr ,
                   0             Gsdy2 1 1d
where exponentially small corrections in N have been
neglected.
  Recognizing this integral (up to a change of variable
and further exponentially small corrections in N) as a beta
function, we find that
                 GsNd    Gsdy2 1 1d1yd Gsk 1 1ydd               FIG. 1. Finite size dependence of rescaled Euclidean 2D
Dk sN, dd ,                  p                    .             TSP optimum. Best fit sx 2 ­ 5.48d is given by LE yN 1y2 f1 1
              GsN 1 1ydd       p          Gskd                  1ys8Nd 1 · · ·g ­ 0.7120s1 2 0.0171yN 2 1.048yN 2 d. Error
                                                                bars represent statistical errors.
                                                      (1)
Notice that there is a complete separation here of the
N dependence and the k dependence. This is indeed               in twenty random starts. These methods introduce a
a surprising universality: It means that up to exponen-         systematic error, because they do not always find the
tially small corrections, all kth nearest neighbor mean         true optimum; we estimated this error by performing a
distances have exactly the same scaling law in N, namely,       large number of runs on a few instances and measuring
GsNdyGsN 1 1ydd. It might be expected, then, that the           the average expected error (weighted by the probability
length of a TSP tour consisting of N links would have           of making that error when choosing the best out of
large N scaling behavior                                        ten random starts). In all cases, we verified that the
        GsNd                                                    systematic error stayed under 10% of the statistical error
 N                 ­ N 121yd                                    shown in the error bars.
    GsN 1 1ydd                                                     In order to reduce the statistical noise further, we used
                         ∑                          µ ∂∏
                               1yd 2 1yd 2             1        the following variance reduction method: Recognizing
                       3 11                   1O 2 ,            that LB sNd ; NsD1 1 D2 dy2 is a lower bound on the
                                    2N                N
                                                                tour length (each city is at best connected to its first-
where the right-hand side follows from Stirling’s formula.
                                                                and second-nearest neighbors), write the estimator for LE
   In fact, due to correlations between k and N in the
                                                                as kL̃E 2 lL̃B l 1 lLB . L̃E and L̃B denotes values for
optimal tour, this is not quite the case. Figure 1 shows
                                                                a particular instance, the angular brackets represent the
our results for LE divided by the scaling quantity above,
                                                                average over instances sample, and the ensemble average
at d ­ 2: We find that this is, to a good fit, itself a power
                                                                LB can be calculated analytically [see Eq. (1)]. l is a
series in 1yN, albeit one with a small first-order term.
                                                                parameter which we adjust to minimize the variance of
The asymptotic N ! ` value is bE ­ 0.7120 6 0.0002,
                                                                our new estimator. In practice, optimal values of l sl ø
where the error is obtained on the basis of x 2 analysis.
                                                                0.75d enabled us to reduce the error by over 60%. Other
This result is, to our knowledge, the most precise to date
                                                                variance reduction methods can also be used [9], but ours
for the Euclidean TSP in the thermodynamic limit.
                                                                has the advantage of introducing no new systematic error.
   The methods by which we obtained the results in Fig. 1
                                                                   Mean-field method.—We now turn our attention to the
are themselves of some importance. For runs optimized
                                                                mean-field approximation, based on the random link TSP.
by LK (N ­ 12 through N ­ 17), we averaged over
                                                                Rather than having N cities distributed randomly in a
the results of 250 000 instances, where for each instance
                                                                hypercube, we now have lengths lij between cities i and
we took the best (lowest) optimum found in ten random
                                                                j s1 # i , j # Nd distributed as independent random
starts (ten different runs). For N ­ 30 we averaged over
                                                                variables according to a certain distribution rsld. We
10 000 instances, taking for each one the best optimum
                                                                take rsld to be the probability distribution of lengths
found by CLO (ten Monte Carlo iterations per run) in
                                                                between cities in the d-dimensional Euclidean problem,
five random starts. For N ­ 100 we averaged over
                                                                in the absence of finite size effects:
6000 instances, taking for each one the best optimum
found by CLO (ten Monte Carlo iterations per run)                           rsld ­ dp dy2 l d21 yGsdy2 1 1d .
                                                                                                                       1189


--- PAGE BREAK ---

VOLUME 76, NUMBER 8                    PHYSICAL REVIEW LETTERS                                           19 FEBRUARY 1996

This establishes a mapping in the thermodynamic limit          TABLE I. Comparison of Euclidean and mean-field TSP
between the random link TSP and the Euclidean TSP,             optima (rescaled) at dimension up to d ­ 3.
neglecting all correlations among (Euclidean) distances.       d               bE                bMF          MF % excess
   The mean-field “model” is the random link TSP,
                                                               1                1               1.0208           12.1%
described for our purposes by the “cavity equations”           2         0.7120 6 0.0002        0.7251           11.8%
written down by Krauth and Mézard [3]. In our language         3         0.6979 6 0.0002        0.7100           11.7%
this leads to
                     ∑             ∏
               1 d Gsdy2 1 1d 1yd
 bMF sdd ­ p
                p 2      Gsd 1 1d
                Z `                                               Figure 2 shows that this is indeed so for the mean-
              3      Gd21 sxd f1 1 Gd21 sxdge2Gd21 sxd dx ,                       p by numerical resolution of Eq. (2).
                                                               field results obtained
                 2`                                            Looking at bMF y dy2pe spdd1y2d , we find an excellent
where bMF , LMF yN 121yd as in the Euclidean case, and         fit by a 1yd power series with a leading order term
           Z ` sx 1 ydd                                        which, to the precision of our raw numerical data, is
  Gd sxd ­              f1 1 Gd s ydge2Gd s yd dy . (2)        indistinguishable from 1.p
            2x    d!                                              The fact that bMF y dy2pe at d ! ` is another
It has been argued persuasively, notably on the basis of       confirmation of the validity of the cavity method, as this
excellent agreement in the d ­ 1 case [3], that the cavity     property is known to be true for the pure random link
method is exact for the N ! ` random link TSP. In              TSP [10]. We have thus added to Krauth and Mézard’s
the following discussion we shall also present further         investigation (at d ­ 1) further evidence (at d ! `) that
justification for this assumption.                             the cavity method is exact.
   There is no known analytical solution of the integral          Finally, let us rewrite the left-hand side of the best-fit
equation for Gd sxd given in Eq. (2). However, it can be       equation in Fig. 2 with an additional s1y2d1y2d factor in
solved numerically; this was done by Krauth and Mézard         the denominator:
at d ­ 1 and d ­ 2, giving bMF sd ­ 1d ­ 1.0208 and                         bMF                       0.499395
bMF sd ­ 2d ­ 0.7251 [3]. These values may be com-                  p                    ­ 0.999997 1
                                                                        dy2pe spdy2d1y2d                  d
pared with bE sdd: Under periodic boundary conditions                                          µ ∂
bE sd ­ 1d ­ 1 (trivially) and bE sd ­ 2d ­ 0.7120 (see                                          1
                                                                                           1O 2 .
previous section). Therefore, at d ­ 1 mean field has a                                          d
2.1% excess with respect to the Euclidean value, and at        Notice that the 1yd coefficient is practically indistinguish-
d ­ 2 a 1.8% excess (see also Table I). Already at low         able from 1y2. An interpretation of this remarkable result
dimension, then, mean field gives quite a good approxi-        is given in [11].
mation to the Euclidean case. It is amusing to note that
Krauth and Mézard themselves assumed a rather inaccurate
Euclidean value bE sd ­ 2d ­ 0.749, and so their mean-
field results seemed poorer to them than they actually were.
   We now extend the numerical solution of Eq. (2) to
higher dimensions. As in the problem of Euclidean
finite size scaling, we can get an indication of what
dimensional dependence to expect in LMF sdd by looking
at the mean kth nearest-neighbor distance Dk multiplied
by the number of links N. In the thermodynamic limit,
Eq. (1) gives 8
                 >
                 >          Gsdy2 1 1d1yd
                 >
                 >  N 121yd      p
                 >
                 >                  p
                 >
                 >
                 >
                 >        Gsk  1 1ydd
                 < 3                   ,
     NDk sdd ,              r Gskd            at large d .
                 >
                 >             d
                 >
                 >  N 121yd        spdd  1y2d
                 >
                 >       ∑ 2peµ         ∂∏
                 >
                 >
                 >
                 >                 lnk
                 :     3  1 1  O
                                    d

Dividing by N 121yd , this suggests that                       FIG. 2. Dimensional dependence of rescaled mean-field
                s                   ∑     µ ∂∏                 TSP optimum.               sx 2 ­ 7.46 3 10211 d is given
                                                                             p Best fit 1y2d
                     d                     1                   by      bMF y dy2pe spdd      ­ 0.999997 1 0.152821yd 1
        bsdd ­            spdd 1y2d
                                      11O      .
                   2pe                     d                   1.05488yd 2 .
1190


--- PAGE BREAK ---

VOLUME 76, NUMBER 8                   PHYSICAL REVIEW LETTERS                                          19 FEBRUARY 1996

                                                              In the process we have extracted what we believe to
                                                              be the best result to date for the thermodynamic limit:
                                                              bE sd ­ 2d ­ 0.7120 6 0.0002.
                                                                 At the same time we have, by means of a mean-field
                                                              method, examined the dimensional dependence of the TSP.
                                                              We have found that mean field is a good approximation (,
                                                              2.1% error) to the Euclidean TSP at d ­ 1, 2, and 3. We
                                                              have seen numerically that at d ! ` the cavity equations
                                                              are compatible with the exact random link TSP result,
                                                              and thus have provided further evidence that they are
                                                              exact at all dimensions. Additional work is in progress
                                                              to understand the coefficient 1y2 in the subleading term
                                                              of the cavity equation solution. Finally, comparing our
                                                              mean-field and Euclidean results suggests not only that the
                                                              Bertsimas-van Ryzin conjecture for the large d limit of
                                                              bE sdd is correct,pbut also that the asymptotic behavior is
                                                              in fact bE sdd ­ dy2pe spdd1y2d f1 1 Os1yddg.
                                                                 We are grateful to N. Cerf for his contributions, as
FIG. 3. Rescaled Euclidean TSP optimum (points) as a func-
                                                              well as to O. Bohigas for having suggested the present
tion of dimension, sandwiched between mean-field optimum      research. We also acknowledge fruitful discussions with
(solid line) and exact lower bound (dashed line).             E. Bogomolny, M. Mézard, S. Otto, and N. Sourlas.
                                                              O. C. M. acknowledges support from NATO travel Grant
                                                              No. CRG 920831. Division de Physique Théoriques is a
                                                              unité de Recherche des Universités Paris XI et Paris VI
   Euclidean model: Dimensional dependence.—Given             associée au CNRS.
the mean-field results, we now return to the Euclidean           Note added.–Since our submission, D. S. Johnson et al.
model. Table I shows the numerical result at d ­ 3            [13], using slightly different methods, have found values
(obtained by the same heuristic methods as in the d ­ 2       for bE sdd compatible with ours at d ­ 2 and 3.
case) together with the mean-field value, and the d ­ 1
and d ­ 2 results presented earlier.
   These results suggest that bMF is an upper bound for
bE (and heuristic arguments [11] also provide support for
                                                                  *Electronic address: percus@ipncls.in2p3.fr
this). At the same time, there is the strict lower bound          †
                                                                    Electronic address: martin_o@ipncls.in2p3.fr
LB ; NsD1 1 D2 dy2, mentioned earlier in the discussion        [1] M. Mézard and G. Parisi, Europhys. Lett. 2, 913 (1986).
on variance reduction. Figure 3 shows the Euclidean re-        [2] M. Mézard, G. Parisi, and M. A. Virasoro, Spin Glass
sults “sandwiched” between the corresponding mean field            Theory and Beyond (World Scientific, Singapore, 1987).
and lower bound quantities, both ofp which may be writ-        [3] W. Krauth and M. Mézard, Europhys. Lett. 8, 213 (1989).
ten in the d ! ` limit as bsdd ­ dy2pe spdd1y2d f1 1           [4] S. Lin and B. Kernighan, Oper. Res. 21, 498 (1973).
Os1yddg. We conjecture that mean field does indeed re-         [5] O. Martin, S. W. Otto, and E. W. Felten, Oper. Res. Lett.
main an upper bound at all values of d, and consequently           11, 219 (1992).
that bE behaves asymptotically at large d as                   [6] J. Beardwood, J. H. Halton, and J. M. Hammersley, Proc.
                  s                                                Cambridge Philos. Soc. 55, 299 (1959).
                                  ∑       µ ∂∏                 [7] D. S. Johnson, in Proceedings of the 17th Colloquium on
                     d                      1
         bE sdd ­        spdd1y2d 1 1 O         .                  Automata, Language, and Programming (Springer-Verlag,
                    2pe                     d                      Berlin, 1990), p. 446.
                                                               [8] W. D. Smith, Ph.D. thesis, Princeton University, 1989.
This would also support a weaker conjecture by Bertsimas       [9] N. Sourlas, Europhys. Lett. 2, 919 (1986).
and van p Ryzin [12], stating that for the Euclidean TSP,     [10] J. Vannimenus and M. Mézard, J. Phys. Lett. (Paris) 45,
bE , dy2pe at d ! `.                                               1145 (1984).
  In conclusion, we have investigated the finite size         [11] N. J. Cerf, J. H. Boutet de Monvel, O. Bohigas, O. C.
behavior of the Euclidean TSP optimum under periodic               Martin, and A. G. Percus (to be published).
boundary conditions, and have seen that at d ­ 2, LE          [12] D. J. Bersimas and G. van Ryzin, Oper. Res. Lett. 9, 223
converges as a 1yN series:                                          (1990).
                                   µ                    ∂     [13] D. S. Johnson, L. A. McGeoch, and E. E. Rothberg, in
             LE                          0.0171                     Proceedings of the 7th Annual ACM-SIAM Symposium
                             ­  bE   1 2         2 · · ·  .
  N 1y2 f1 1 1ys8Nd 1 · · ·g                N                       on Discrete Algorithms (to be published).




                                                                                                                      1191


--- PAGE BREAK ---


