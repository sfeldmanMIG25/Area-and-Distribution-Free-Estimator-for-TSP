# An asymptotic determination of the minimum spanning tree and minimum matching constants in geometrical probability

> Citation key: `bertsimas1990asymptotic`  
> DOI: 10.1016/0167-6377(90)90065-F  
> Download URL: https://web.mit.edu/dbertsim/www/papers/AppliedProbability/An%20asymptotic%20determination%20of%20the%20minimum%20spanning%20tree%20and%20minimum%20matching%20constants%20in%20geometrical%20probability.pdf  
> SHA-256: `6cd3e8b0bc082fd9cf901becf52ce77ac16dee1ee5f0e090a15cc3d527e10686`  

---

         An Asymptotic Determination of the Minimum Spanning
         Tree and Minimum Matching Constants in Geometrical
                                               Probability

                            Dimitris J. Bertsimas *           Garrett van Ryzin t

                                               August 1, 1989



                                                   Abstract

                    Given n uniformly and independently distributed points in a ball of unit
                 volume in dimension d, it is well established that the length of several combi-
                 natorial optimization problems (including the minimum spanning tree (MST),
                 the minimum matching (M), the traveling salesman problem (TSP), etc.) on
                 these n points is asymptotic to P(d) n(d- 1)/d, where the constant B(d) depends
                 on the dimension d and the problem solved. It has been a long open problem
                 to determine the constants 38(d) for these problems. In this paper we make
                 progress in establishing the constants PMST(d), 3M(d) for the MST and the
                 matching problem. By applying Crofton's method, an old method in geomet-
                 rical probability, we prove that PMST(d)     /,     6IM(d)           as d tends
                 to infinity. Moreover, our method corresponds to heuristics for these problems,
                 which are asymptotically exact as the dimension increases. Finally, we examine
                 the asymptotics for the TSP constant and improve the best known bounds for
                 d= 3,4.
           · Dimitris Bertsimas, Sloan School of Management and Operations Research Center, MIT, Rm
         E53-359, Cambridge, MA 02139.
           t
               Garrett van Ryzin, Operations Research Center, MIT, Cambridge, MA 02139.


                                                        1




C3   _                __   __I_


--- PAGE BREAK ---

                                                      1l




   Key words. Probabilistic Analysis, Euclidean Spaces, Crofton's Method, Mini-
mum Spanning Tree, Matching.




                                       2




                                            ---
                                          ----
                                     ---------                                         __X____
                                                                                  _1·1__


--- PAGE BREAK ---

     1        Introduction

     Research in the area of probabilistic analysis of combinatorial optimization problems
     in Euclidean spaces was initiated by the pioneering paper by Beardwood, Halton
     and Hammersley [1], where the authors prove the following remarkable result:
     Theorem ([1]): If Xi are independent and uniformly distributed points in a region
     of Rd with volume a, then the length LTSP of the traveling salesman tour (TSP)
     under the usual Euclidean metric through the points X 1 ,..., X, almost surely sat-
     isfies
                                     lim   LTSP           Tsp(d)a/d,
                                    n--oo n(d-1)/d -

     where PTsp(d) is a constant that depends only on the dimension d.
             This result was generalized to other combinatorial problems defined on Euclidean
     spaces, including the minimum spanning tree (MST) ([9]), the minimum matching
     (M) ([6]), the Steiner tree (ST) ([8]), the Held and Karp (HK) lower bound for the
     TSP ([4]) and other problems. Indeed, Steele [8] generalized the previous theorem
     for a class of combinatorial problems called subadditive Euclidean functionals. All
     these results say that there exist constants         3 MST(d),,3M(d),,   ST(d), PHK(d) such

     that the corresponding values for the MST, M, ST, HK over n(d-l)/d tend almost
     surely to the corresponding constants. One of the important open problems in this
     area is the exact determination of these constants.
             In this paper we make progress for two of these constants, namely PMsT(d), PM(d).
     We find upper and lower bounds for the MST(d), PM(d) that are asymptotic to the
     same value as the dimension d increases. As a result, we prove that

                                                      dT1              d
                                PMST(d)      ~    ,    P3M(d)~         2- e

     We use Crofton-'s method, an old but surprisingly neglected method in geometrical
     probability, to compute the upper bounds. The bounds correspond to heuristics for
     these problems which have the interesting property that they are asymptotically



                                                      3




I   ______


--- PAGE BREAK ---

exact as both the dimension and the number of points increases. Furthermore, the
lower bounds are based only on elementary arguments concerning nearest neighbors.
    The paper is structured as follows. In the next section we find lower bounds for
the two constants. In Section 3 we review Crofton's method and use it for the MST
and the minimum matching problem in order to find upper bounds for fMST(d) and
#iM(d). In addition, we analyze the connection of Crofton's method and heuristics
for the MST and the matching problem. In Section 4 we state our central results on
the asymptotics of PMST(d) and PM(d) and extend the results for more general MST
functionals studied by Steele [9]. Finally, in Section 5 we review the best bounds for
the Tsp(d), and observe that bounds based on our results improve the best known
bounds for d = 3,4. We also conjecture that

                                 Prs(d)         d
                                               27re

2       Lower bounds

The lower bounds on     MST(d),f M(d) are found from the following elementary ob-
servation. The MST has larger length than the sum of the lengths of the nearest
neighbor from each point. If L,(a), M,(a) are the expected length of the MST
and minimum matching on n uniformly and independently distributed points in the
d-ball of volume a , then

                             Ln(a) > (n - 1)N(n, d, a),                            (1)

where N(n, d, a) is the expected length of the nearest neighbor of a random point
in the region. Similarly, since in every matching there are n/2 (assume n is even)
edges
                                         n
                               Mn(a) > N(n, d a).                                  (2)
In the following lemma, which is well known, we find a lower bound on N(n, d, a).
Based on this bound Beardwood et. al. [1] find a lower bound for #Trsp(d), although
they did not include its proof. We present the proof for completeness.

                                          4




                                                   ~ ~ ~ ~ ~ ~~~ _
                                          __---__II(~~~~~~~___       _ r~~~il__l*___n~~~~~~~^~~_r________~~~~~


--- PAGE BREAK ---

Lemma 1
                                   N(, d,)          > d(-                 r(),                            (3)

where Cd =    ir)       is the volume of the ball of unit radius in dimension d.

Proof
If d(p) denotes the length of the nearest neighbor to a point p, then

                                 Pr{d(p) > r} = (1-(P'r))
                                                                      a

where V(p, r) is the volume of the intersection of the given d-ball of volume a and
the ball of center p and radius r. Thus V(p, r) < min(cdrd, a). As a result,
                        (_ &da                                      X)l/d1(

    N(n, d, a) =       '"         Pr{d(p) > r}dr > A                          (1 - min(cdrd/a, 1))n-dr.

If we make a change of variables to z = cdrd/a, we obtain

                 N(n,d,a)          >           /d
                                               A       zl/d         (1-min(z,))a-ldz

                                            dc I
                                   >--    d/           zl/d-(l - z)n-Xdx



                                         dc/d r(n +
                                          1 (da        7




where the last two relations follow from well known properties of the gamma function
(see for example Rudin [7], p.192).                                                                        0
   From (1), (2), (3), and since limn-.,o                   dz       = I3MsT(d)al/d, limn--.OO             =

l3M(d)al/d, we find the following lower bounds on fIMST(d),/lM(d):
                                                                1         1
                                         PMsT(d) > -                r( ),                                 (4)

                                         PM(d) >           dc          )'                                 (5)

               d=
                l2dc
              where          Cd
where   d=     dr(
             F(?1_+1

                                                        5




               _____


--- PAGE BREAK ---

3      Crofton's Method Applied to MST and MATCH-
       ING

Crofton's Method is an old and specialized technique for determining mean values
of random variables in certain geometrical probability problems (see for example
[5]). It applies to problems in which n points and independently and uniformly
distributed in a region of the d dimensional space and one would like to determine
the mean value of some function defined on these points. The fundamental idea
of the method is to view the expectation as a function of the size of the geometric
region and then to derive a differential equation or this function by perturbing the
region's size. In our case, the function is the length Of the MST (or the matching
problem) of n such points and the region is a d-dimensional ball of volume a centered
at the origin.
      Let L,(a) denote the expected value of the MST as a function of n and a.
Suppose we increase the volume of the ball incrementally from a to a + ba and
distribute n points uniformly and independently in the enlarged ball. We consider
two events:

      * E: The event all n points lie in original ball of volume a.

      * E 2: Exactly n - 1 points lie in the original ball of volume a and one point lies
        in the infinitesimal, spherical shell of volume 6a.

All other events have probability o(6a) and are ignored, since we will take a -, 0.
Note that

                         P(E1) = [a               = 1-   a + o(6a),                   (6)
and
                                   6a     [     a n-i   n
                     P(E 2 )= n                -      = -6a + o(6a).                  (7)
                                  a+a         a+arged a
The expected length of the MST in the enlarged ball given E 1 is just Ln(a). If we



                                                  6




                                   __ _1_1_1___


--- PAGE BREAK ---

let L,(aIE2 ) denote the expected length of the MST given E 2, then

                L(a + 6a) = L(a)P(El) + L.(aIE2 )P(E2 ) + o(ba),

which using (6) and (7) gives

               L(a +    a) - L(a)+          L(a) =       Ln(aIE 2 ) +
                        6a              a            a                  6a

Letting 6a -, 0, we obtain

                            dL(a) + L(a) = Ln(aIE 2).                               (8)
                             da    a      a

   Observe that we can construct a feasible spanning for the case where one point
lies on the exterior of the ball (event E 2 ) by forming the MST of the n - 1 points in
the interior of the ball and then connecting the exterior point to the closest point on
this tree. If we let Rd(n, a) denote the expected value of the distance from a point
on the exterior of the d-dimensional ball of volume a to the nearest of n uniform
points in the interior, then

                           Ln(alE2) < Ln_l(a) + Rd(n- 1, a).

Substituting this into (8) gives

                  dLn(a)       n        n                 n
                     d-- )+ -Ln    < Ln-l(a) +- Rd(n - 1, a).                       (9)
                     da      a (a)  a          a

Note that if we defined Ln(a) has the expected length of the spanning tree that
results from inserting points according to their distance from the origin, i.e. the
ezodic tree heuristic (see Section 3.1), then we could express (9) as an equation
rather than an inequality; however, since we are ultimately concerned with the
MST, we choose to work with the MST length and the resultant inequality (9)
directly.
    We are naturally led to find an upper bound on Rd(n, a).




                                             7


--- PAGE BREAK ---

Lemma 2

               1 (2a )         r() < R(n, a) <                         2 a-)        r(    )+         (6),                (10)

where cd = (+1     is the volume of the ball of unit radius in dimension d and 6 is a
constant such that 0 < 6 < 1.

Proof
The complement of the distribution of the distance from a point on the exterior of
the ball to the nearest point is
                                                                               2a
                                (1- A(r)/a)n,                     0 < r < 2/
                                                                               Cd
where A(r) is the volume of the intersection of a ball of radius r centered at the
exterior of the ball of volume a with the interior of the ball of volume a. Then
                                                    ( 2a)l/d

                           Rd(n,a) =                Cd         (1 - A(r)/a)ndr.

Intuitively the asymptotically important contribution in the above integral comes
near r = 0, where A(r)              Ed. In order to formalize this we make the following
observation. For every b < 2 there exists a , such that


                           A(r) Ž              bcdd
                                                                  < r < (2a)1/d

Therefore,

             Rd(n,a)       <           (1--)               dr +           cd         (1        bcd      nd
                                                    a                               (1                 )dr
                                    IV_      bncdrd



                           <    a              ae       dr + 0(6),


where 0 < 6 = 1 - b'd < 1. If we make a change of variables z =                                             cdrd, then
                       a                                                                                     a

                 Rd(n,a) < - ( a                              |        l/dledz + O()

                                =              a         l/dr()+o(6'n).
                                      d      bCdn                  d

                                                          8




                                    ------                                     -C·c"l"----·zunari.FII               --------in-ixr·immxraanra-sm^rr


--- PAGE BREAK ---

Since we can select any b <             the right inequality in (10) follows.
                    f d) we can use exactly the same method as in Lemma 1 to
Since A(r) < min(a, d-
prove the left inequality in (10).                                                        0
      We next characterize the solution to the differential-difference equation (9) as
n -    oo. Upon substituting the results of Lemma 2 into (9) we note that the error

term is O(nbn), i.e. exponentially small, and therefore can be neglected for large n.
By a simple scaling argument, Ln(a) = Lnal/d, where Ln is the expected length of
the MST in a unit d-ball. Substituting this into (9) does indeed solve the differential
equation and yields the following difference equation for Ln,
                              L,
                              d + nLn < nL,_ 1 + Rdn(n-                  ) - l/d       (11)

where Rd = a ()ld         r().
      To determine the leading behavior of the solution Ln for n -- oo we note that
the corresponding differential equation to (11) is

                                   dy = _y           + Rd(z       ) - l/d,
                                   dx          xTd
which has the following asymptotic behavior as z - oo

                                              y(x)     Rd-.

Because z = oo is a regular singular point for the differential equation, the leading
behaviors of the difference and corresponding differential equations are the same
(see Bender and Orszag [2] pages 64 and 206).
      To verify this we try a solution of the form Ln = flMsT(d)n(d-1)/d and check
that it is consistent with (11) for n -. oo. Substituting Ln = PMsT(d)n(d- 1 )/d into

(11) yields
/3MST(d)n(d-)/d
           (d)n( )/+    PMST(d)n(d- 1)/d+1 < P3MST(d)n(n - l)(d- l )/d + Rdn(n -     ) - 1/d .

Using the expansion

                       (n -    1 )(d-l)/d -    n(dl)/d(       -         + O(n 2)),
                                                                  dn

                                                       9


--- PAGE BREAK ---

and rearranging we obtain

                               PMST(d) < Rd + O(-)-
                                                n

Therefore, the solution is indeed consistent for n -   oo if PMST(d) < Rd. Moreover,
it is also consistent with our a priori knowledge of the asymptotic behavior of the
MST length; therefore, we conclude that
                                            21/dr()                             (12)
                                               dCd
In addition, the leading behavior of the expected length of the exodic tree heuristic
is Rdn(d-l)/dal/d

   With some modification, a similar analysis applies to the minimum matching.
Let Mn(a) be the expected length of the minimum matching. Applying the same
technique to the matching problem we obtain

              dM,(a) + -Mn(a) < -Mn- 2 (a - cR) +             Rd(n- 1, a),       (13)
                 da      a           a                    a

where Mn-2 (a - CR) is the length of the matching of the n - 2 points in the region
consisting of the sphere of volume a minus those points within a radius Ra(n - 1, a)
of the exterior point. Note that within this region, the n -2 point are uniformly and
independently distributed; therefore, since the remaining n- 2 points are uniformly
and independently distributed in an area which is a subset of the original area
Mn(a - cR) <   Ms(a) and since CR -- 0 as n -- oo


                         Mn(a - cR) = Mn(a)            n ~ oo.

Therefore (13) becomes

                 dMn(a)             n
                        + aMn(a) < -Mn-2(a)
                   da d()           a       + na Rd(n - 1, a).                   (14)

Again, it is straightforward to verify that a solution of the form Mnald solves (14)
and yields
                        d " + nMn < nM._ 2 + Rdn(n- 1) - / .                     (15)
                        d

                                          10


--- PAGE BREAK ---

      The only difference between (11) and (15) is that in the matching problem given
      that we match two points there are n- 2 remaining points to be matched, while in
      the MST given that we connect one point there are n - 1 points remaining to be
      connected.
            Following the same analysis as in the MST and using the expansion
                          (n   2) ( d- ) / d = n(d-l)/d(l -   2(d - 1) +   (n   )
                                                                dn

      we obtain
                                         13M(d)        2/r(                              (16)
                                                     (2d - 1)Id
      3.1      Relation of the Crofton method with heuristic algorithms

      Crofton's method produced an upper bound for the the values of the constants
      J3MST(d),PIM(d). As mentioned, the method gives rise to heuristic algorithms for
      the MST and the matching problem. For the MST problem the heuristic algorithm
      is the following:

         1. Given the points Xl,...,X, sort the points such that Xli < X 2 1 < ... C
              IXnl.


         2. For i = 2, 3,..., n connect Xi to Xj , such that lX, - Xl = min<i IXi - XI.
              It is easy to see by induction orL i that these n - 1 connections form a tree.

      This construction was proposed by Gilbert [3], who gave it the name the exodic tree
      heuristic. He obtained the same constant (12) for the expected value of the ex-
      odic tree for d = 2 through the use of generating functions and multidimensional
      integrals. Our analysis using the differential equation approach is considerably eas-
      ier, has the advantage of naturally generalizing to every d and also generalizes to
      matching and possibly to other problems.
            A similar connection holds for the matching problem. Namely our construction
      gives rise to a heuristic algorithm, which we call exodic matching to parallel the
       MST construction, as follows:

                                                       11




__I                _


--- PAGE BREAK ---

    1. Given the points X,...,Xn sort the points such that IXl <I X 2 l < ... <

        IXnl.


    2. Starting with the outer point X, connect X, to its nearest neighbor, call it Xj.
        Delete X., Xj from the list of points. Repeat the procedure for the remaining
        points thus producing a matching.

    As we show in the next section these heuristics have the surprising property of
being asymptotically optimal as the dimension d increases. Furthermore, we find
simple asymptotic values for JMST(d), fM(d) as the dimension increases.


4     The main result

In this section we combine the upper and lower bounds derived in the two previous
sections and derive the central result of the paper.

Theorem 3 The constants IMST(d),# M(d) satisfy

           1r(   + 1)r( + )1/d < MST(d) <                               F(   + 1)r( + l)1/d 2 1/d, and

      1   1            d                      d      1                          1            d
               )
     2VFr(d +1)(             1 I/d< tM(d) < 2d - 1r(                                + 1)r(       + )d1) 2 1/d
As d -- oo

                           tMsT(d)                       e       M(d)          2     .                          (17)

Proof
The bounds for OtMST(d) follow by combining (4) with (12) and (5) with (16). Note
that r(z+ 1) = zr(z). (17) follows from the fact that limd..o 2i = 1, limd--o,,                             da =
1 and the asymptotics of the gamma function, i.e. r(k + 1)                                        v/2k+2e - k as
k -- oo.                                                                                                           0
     From Theorem 3, we remark that for large dimensions almost every point in the
optimum MST and matching is connected to its nearest neighbor, since we derived



                                                             12




                             _____l____i_________srr___·nx__l____rr_·                                                  L-----^-X-·--


--- PAGE BREAK ---

     the lower bound from exactly this observation. Moreover, the value of the MST is
     twice the value of the matching problem.
            Although we only determined the two constants asymptotically for large dimen-
     sions, the upper and lower bounds are close to each other even for small dimensions.
     In Table 1 we calculated the upper and lower bounds and the asymptotic values
     from Theorem 3 for          MST(d) as a function of d.


                         d        Lower bound     Upper bound           Asymptotics
                          10      0.866331        0.928511              0.765179
                          50      1.77978         1.80462               1.71099
                          100     2.47619         2.49342               2.41971
                          150     3.01347         3.02743               2.96352
                          200     3.46761         3.47965               3.42198
                          300     4.23107         4.24085               4.19106
                          400     4.87577         4.88422               4.83941
                          500     5.44433         5.45188               5.41063
                          1000    7.67823         7.68355               7.65179

               Table 1: The bounds and the asymptotics for PMST(d) as a function of d.


         4.1    Generalizations to more general MST functionals

         Steele [9] analyzed the asymptotic behavior of the following MST functional. Let
         O(2) be a monotone function and Sn = minT CeET (je ), where the minimization
         is taken over all spanning trees. Note that when +(z) = z the problem reduces to
         the MST and also, since +(z) is monotone, the MST is the optimal tree for every
          b(z). Furthermore, he proved that if +(z) - zk, k < d as z -, 0 and the points
         are uniformly and independently distributed in a d ball of volume 1, then with
         probability 1
                                          lim n(d-k)/d
                                         n..oo  S            3(d, k).

                                                    13




-I----           ---                         --


--- PAGE BREAK ---

In particular P(d, 1) = flMST(d). By applying exactly the same methods for the
upper and lower bounds we can establish the following bounds for fP(d, k):

                           k     ()<     k          k kd
                         d k/r()     < (d,k) <•7/dr()k/d.                                 (18)

As a result, by examining the asymptotics of the upper and lower bounds we can
easily obtain that for large d

                                      (k, d)        d k/2
                                                   2re


5    The best known bounds for the TSP

In this section we summarize the best known analytic bounds for the constant
PfTsp(d) for the TSP. The goal of this section is to compare the bounds and to
reveal that this constant has also an E(vd) behavior, which is consistent with the
behavior of PMST(d) and         M(d) from the previous section.

Theorem 4 P3Tsp(d) satisfies

            1     1       1                       'f2
            1r(       )( + 1       PTSP(d) < min[2a    1
                                                      -r(d),               f121/,
         dc Ild
            d
                           2d                  -              d
                                                                   _

where cd = Fgy Asymptotically, PTsp(d) = e(vi).

Proof
The lower bound follows by considering both the nearest and the second nearest
neighbor for each point, since in the optimal tour there are two edges coming out
from each point, whose length should be greater than the sum of the two closest
neighbors. The upper bound follows from [1] by applying the strip heuristic and
from the observation that twice the length of the MST is an upper bound of the
optimum TSP tour, where we used the upper bound for the MST from (12). In fact,
in [1] the authors prove a better bound for d = 2, i.e.           rTSp( 2 ) < 0.9204. Note that



                                               14




                                          I II^ _      -- _


--- PAGE BREAK ---

for d = 3,4 the bound from the MST is better than the one of [1] and for d > 5 the
bound of [1] dominates. Asymptotically, for large dimensions

                                    d
                                        < Tsp(d) < X,

i.e. PTsp(d) = e(Vrd).                                                          °
    In table 2 below we compute the best known upper and lower bounds for the
TSP constant for dimensions d = 2,..., 10.


                           d     Lower bound     Upper bound
                           2     0.625000        0.92040
                           3     0.646287        1.39589
                           4     0.684158        1.44641
                           5     0.724529        1.50053
                           6     0.764356        1.51309
                           7     0.802857        1.54043
                           8     0.839871        1.57531
                           9     0.875436        1.61419
                            10   0.909648        1.65517

                      Table 2: The best known bounds for PTsp(d).


6    A closing conjecture

Our success with the MST and the matching constants, at least asymptotically,
raises the natural question whether the upper bounding method can work for other
combinatorial problems, in particular the TSP. Despite our attempts we were not
able to generalize the method for the TSP. In light of Theorem 3 and our prelimi-
nary -calculations we conjecture that the asymptotic behavior for large dimensions
of l'Tsp(d) is   /,     which is consistent with the statements in [1].

                                            15


--- PAGE BREAK ---

Acknowledgment
The research of both authors was partially supported by the National Science Foun-
dation under grant ECS-8717970.


References

 [1] J. Beardwood, J.H. Halton and J.M. Hammersley,                                 The shortest path through
    many points", Proc. Cambridge Philos. Soc., 55, 299-327 (1959).

 [2] C.M. Bender and S.A. Orszag, Advanced Mathematical Methods for Scientists
    and Engineers, McGraw-Hill, New York, 1978.

 [3] E.N. Gilbert, "Random minimal trees", J. Soc. Indust. Appl. Math., 13, 376-387
    (1965).

 [4] M.X. Goemans and D. J. Bertsimas, "Probabilistic analysis of the Held-Karp
    relaxation for the Euclidean traveling salesman problem", MIT Sloan School
    Working Paper (1988), to appear in Mathematics of Operations Research.

 [5] R. Larson and A. Odoni, Urban Operations Research, Prentice-Hall, New Jersey
    (1981).

 [6] C.H. Papadimitriou, "The probabilistic analysis of matching heuristics", Proc.
    15th Annual Conference Comm. Contr. Computing, Univ. Illinois (1978).

 [7] W. Rudin, Principles of Mathematical Analysis, third edition, McGraw-Hill,
    Tokyo (1976).

 [8] J.M. Steele, "Subadditive Euclidean Functionals and Nonlinear Growth in Ge-
    ometric Probability", Ann. Prob., 9, 365-376 (1981).

 [9] J.M. Steele, "Growth Rates of Euclidean Minimal Spanning Trees with Power
    Weighted Edges", Princeton University, technical report (1988).


                                                    16




                                  ------------ ·- ^-II   ^·axorrrarxrrra*1CJ--.XI.I.-..F---                               I1..-.--·-
                                                                                                                -11·-1_-.-_-      I------__C·lti-_lllII-


--- PAGE BREAK ---


