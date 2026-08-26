# A fast and scalable radiation hybrid map construction and integration strategy

> Citation key: `agarwala2000fast`  
> DOI: n/a  
> Download URL: https://europepmc.org/api/getPdf?pmcid=PMC311427  
> SHA-256: `e942b8c8f02fbeaf87530c135286981ef441b4fff598b27ae79a88965937012d`  

---

Methods


A Fast and Scalable Radiation Hybrid Map
Construction and Integration Strategy
Richa Agarwala,1 David L. Applegate,2 Donna Maglott,1 Gregory D. Schuler,1
and Alejandro A. Schäffer1,3
1
 National Center for Biotechnology Information (NCBI), National Institutes of Health (NIH), Bethesda, Maryland 20894
USA; 2Department of Computational and Applied Mathematics, Rice University, Houston, Texas 77005-1892 USA


      This paper describes a fast and scalable strategy for constructing a radiation hybrid (RH) map from data on
      different RH panels. The maps on each panel are then integrated to produce a single RH map for the genome.
      Recurring problems in using maps from several sources are that the maps use different markers, the maps do
      not place the overlapping markers in same order, and the objective functions for map quality are incomparable.
      We use methods from combinatorial optimization to develop a strategy that addresses these issues. We show
      that by the standard objective functions of obligate chromosome breaks and maximum likelihood, software for
      the traveling salesman problem produces RH maps with better quality much more quickly than using software
      specifically tailored for RH mapping. We use known algorithms for the longest common subsequence problem
      as part of our map integration strategy. We demonstrate our methods by reconstructing and integrating maps
      for markers typed on the Genebridge 4 (GB4) and the Stanford G3 panels publicly available from the RH
      database. We compare map quality of our integrated map with published maps for GB4 panel and G3 panel by
      considering whether markers occur in the same order on a map and in DNA sequence contigs submitted to
      GenBank. We find that all of the maps are inconsistent with the sequence data for at least 50% of the contigs,
      but our integrated maps are more consistent. The map integration strategy not only scales to multiple RH maps
      but also to any maps that have comparable criteria for measuring map quality. Our software improves on
      current technology for doing RH mapping in areas of computation time and algorithms for considering a large
      number of markers for mapping. The essential impediments to producing dense high-quality RH maps are data
      quality and panel size, not computation.

Many genome-wide maps have been constructed as                       the current maps and the new maps we compute are
part of the Human Genome Project. A current widely                   consistent with human DNA sequence contigs in Gen-
used technique is radiation hybrid (RH) mapping (Goss                Bank.
and Harris 1975; Cox et al. 1990; Walter et al. 1994).                    It is possible to reconstruct maps of previously
One purpose of constructing maps is to provide land-                 mapped markers because the RH database (RHdb,
marks along each chromosome to guide sequencing of                   http://www.ebi.ac.uk/RHdb/index.html) contains
the DNA. To date, most of the mapping effort has been                publicly submitted RH vectors (rhvectors) for se-
put into iteratively constructing denser and denser                  quence-tagged site (STS) markers. An rhvector for an
maps rather than integrating new maps with old maps.                 STS x is a vector (x1, x2, . . . , xn), where n is the number
Recurring problems in using maps from several sources                of hybrids (or cell lines) in the RH panel and each
are that the maps use different markers, the maps do                 xi = 0, 1, 2, depending on whether hybrid i is typed
not place the overlapping markers in same order, and                 and retains x, typed and does not retain x, or not
the objective functions for map quality are incompa-                 typed and/or ambiguous, respectively (Cox et al. 1990;
rable. Because many large contigs of human DNA se-                   Boehnke et al. 1991; Matise et al. 1998).
quence are now finished and submitted to GenBank, it                      The rhvectors in RHdb are generated from mul-
would be desirable to integrate maps of markers with                 tiple mapping panels; those reviewed in this paper are
the DNA sequence so that the maps can continue to be                 from the Genebridge 4 (GB4) panel (Gyapay et al.
used to fill in the rest of the sequence and to identify             1996) and the Stanford G3 panel (Stewart et al. 1997).
genes in regions bounded by well-mapped markers.                     Previously published maps used the GB4 and G3 pan-
     In this paper we propose and evaluate new strate-               els independently and used independent resources
gies for reconstructing RH maps and integrating those                such as YAC contig data to build their maps (Hudson et
maps as well as others that have comparable objective                al. 1995; Deloukas et al. 1998). We decided to recon-
functions for map quality. We also evaluate whether                  struct the RH maps to take advantage of the fact that
                                                                     some markers were typed on both panels. The concat-
3
 Corresponding author.                                               enation of rhvectors for the same marker from both
E-MAIL schaffer@helix.nih.gov; FAX (301) 480-9241.                   panels makes the resulting rhvectors longer, which


350    Genome Research                   10:350–364 ©2000 by Cold Spring Harbor Laboratory Press ISSN 1088-9051/00 $5.00; www.genome.org
          www.genome.org


--- PAGE BREAK ---

                                                                               Integrating Radiation Hybrid Maps


Ben-Dor and Chor (1997) showed is essential to com-           computing a RH map for either the OCB or MLE crite-
pute more accurate RH maps.                                   rion can be mathematically transformed into an in-
     RH mapping is based on the hypothesis that the           stance of a much studied optimization problem called
closer the loci are on a chromosome, the more likely          the traveling salesman problem (TSP; Karp et al. 1996;
they are to be retained or lost together in a hybrid. That    Ben-Dor and Chor 1997). The transformation employs
is, their rhvectors will have few differences. The two        an approach using multiple pairwise comparisons be-
criteria typically used for assessing the closeness of        tween markers rather than the more commonly used
rhvectors are the number of obligate chromosome               multipoint comparisons. The transformation is exact
breaks (OCB) and maximum likelihood estimate                  when there are no unknown entries in the data and
(MLE). Other criteria like Bayesian posterior probabili-      approximate otherwise. The TSP has been the subject
ties involve more modeling assumptions (Lange et al.          of intense research for decades (Papadimitriou
1995) and have not been used in developing software           and Steiglitz 1982; Lawler et al. 1985; Reinelt 1994),
for computing RH maps. It is known that OCB and               and there is now a superb software package called
MLE are not identical, but to our knowledge, Ben-Dor          CONCORDE (combinatorial optimization and net-
and Chor (1997) are the first to show that OCB and            worked combinatorial optimization research and
MLE are equivalent under conditions of equally spaced         development environment; Applegate et al. 1998)
markers and 50% retention of markers on hybrids.              for solving large instances. We decided to test
However, these conditions are not satisfied by data on        CONCORDE for RH mapping as part of our effort to
current panels. We verify the incomparability of the          reconstruct maps. An unintended result of our experi-
two objective functions.                                      ments is that CONCORDE consistently computes maps
     The number of OCB for a marker order on a RH             with lower OCB and higher MLE than those computed
map with markers typed on the same panel is the num-          by RHMAPPER. Moreover, CONCORDE is much faster
ber of times a 1 is followed by a 0 or vice versa, ignoring   on large data sets than RHMAPPER when RHMAPPER is
intervening 2s (unknown), between consecutive mark-           required to compute its initial framework internally de
ers at all vector positions. The OCB objective for creat-     novo. In the past, the users of RHMAPPER have con-
ing a map from rhvectors, then, is to find the marker         structed an initial framework map, in part, by relying
order that implies the minimum number of OCB                  on information from other sources such as genetic map
among all possible marker orders. For the MLE objec-          and YAC contig data (Slonim et al. 1997).
tive, the breakage probability and retention probability           Ben-Dor and Chor (1997) showed that with the
are calculated from rhvectors that are then used for          current number of hybrids, the probability of getting
estimating the distance between markers and the like-         “the correct order” for all the markers is very low
lihood of a map. The order of the markers that maxi-          (<0.01). Even for only 20 markers, the success probabil-
mizes the likelihood of the map is considered the true        ity is <0.5, so any strategy that is pinned to framework
order of markers on the map.                                  maps of >20 markers is likely to produce maps with
     Current RH maps are produced with specially tai-         serious large-scale errors. The attempts made to model
lored software packages such as RHMAP (Boehnke et al.         errors in data by hidden Markov models (Heath 1997;
1991), RHMAPPER (Slonim et al. 1997), and MultiMap            Slonim et al. 1997) have been successful in placing a
(Matise and Chakravarti 1995). The packages currently         few hundred markers but cannot be used for placing
in use choose either OCB or MLE as the objective func-        the thousands of markers that are becoming available
tion and use statistical parameters and/or heuristics to      without starting from a fairly dense initial framework
produce a map. When using MLE, Lange et al. (1995)            map.
proposed a way of constructing a model that specifi-               For consistency, we compare previous maps and
cally incorporates the possibility of typing error and        our integrated map with large sequence contigs sub-
presence of unknowns, and Lunetta et al. (1996) spe-          mitted to GenBank. The maps are consistent with the
cifically allowed for multiple panels. We propose ex-         sequence if markers are placed in the correct sequence
tensions to the OCB and MLE objective functions, dif-         order on the map. We choose this objective function
ferent from those in previous papers such as (Lange et        for map quality because there is currently no good way
al. 1995), to incorporate the presence of unknowns and        to assess how much better one map is compared with
present a strategy that identifies markers with the same      another one in terms of the number of markers actu-
map order independent of which extended version of            ally ordered correctly except for chromosome 22 for
objective functions is used.                                  which the completed sequence is available. We find
     We borrow several tools and techniques from do-          that all of the maps are inconsistent with the sequence
mains of computer science and combinatorial optimi-           data for at least 50% of the contigs, but our integrated
zation (Papadimitriou and Steiglitz 1982) to design and       maps are more consistent. We provide some evidence
implement our strategy. It has been known for several         that the inconsistencies are in large part due to data
years that for haploid error-free data, the problem of        quality or panel sizes and not as much due to mapping


                                                                                           Genome Research       351
                                                                                             www.genome.org


--- PAGE BREAK ---

Agarwala et al.


strategy. We also list the number of markers in the                 1. The retention probability p of the data set is esti-
same order between every pair of Généthon (Dib et al.                mated by the ratio of the total number of 1s to the
1996), RH Consortium (Deloukas et al. 1998), Stanford                  total number of 1s and 0s.
(Stewart et al. 1997), and our integrated map.                      2. The likelihood of observing rhvector (x1, x2, . . . , xn)
     The next section of this paper presents definitions               for a single marker x is
and theoretical background on RH mapping and its
relationship to problems in combinatorial optimiza-                                   L共x兲 = 关1 − qc兴n1 × 关qcn0兴               (1)
tion. (More background material that is relevant to the
rest of the paper, but less essential, can be found in the             where c is 1 for haploid, 2 for diploid, q = 1 ⳮ p, and
Appendix.) This is followed by a section in Results de-                nj is the number of positions i such that xi = j.
scribing our map reconstruction strategy, our map in-               3. The likelihood of observing rhvectors for a pair of
tegration strategy, and our computational experiments                  markers x and y is
with these strategies. We conclude with a short Discus-                L共x, y兲 = L共x兲L共y | x兲                                  (2)
sion and a short section on Methods summarizing                                = L共y兲L共x | y兲                                  (3)
availability of our software and data.                                         = 共1 − 2qc + 关q共1 − ␪x,yp兲兴c兲n11关qc共1 − 共1
Definitions and Theoretical Background                                           − ␪x,yp兲c兲兴共n01+n10兲关q共1 − ␪x,yp兲兴cn00        (4)
Our methods rely on known algorithms for two prob-
lems widely studied in computer science and combi-                     where ␪x,y is the breakage probability between mark-
natorial optimization: the longest common subse-                       ers x and y, and nij is the number of positions r such
quence problem (LCSP, sometimes also called the long-                  that xr = i and yr = j.
est common substring problem) and the TSP. Both                     4. L(x,y) is maximized when ␪x,y is the smaller root of
LCSP and TSP have many applications to problems in                     the equation obtained by setting the derivative of
computational biology (Gusfield 1997) but may be un-                   L(x,y) with respect to ␪x,y to 0. For the diploid case,
familiar to practitioners of RH mapping. Therefore, we                 the equation to be solved is a degree five polyno-
summarize the most essential background material in                    mial, and for the haploid case, we get a degree two
this section. More background material including a                     polynomial whose solution gives the following:
brief history of the TSP can be found in the Appendix.                 ␪x,y = 关共n − n11p − n00q兲
LCSP                                                                        − 公共n − n11p − n00q兲2 − 4npq共n10 + n01兲兴 Ⲑ 共2npq兲
Given two sequences A = a1, a2, . . . , an and B = b1,                                                                    (5)
b2, . . . , bm, find a longest sequence C = c1, c2, . . . , ck
such that C is a subsequence of both A and B. For                      The root of the quadratic equation chosen for ␪ is
example, if A = a, l, g, o, r, i, t, h, m and B = l, o, g, a, r,       the smaller root to satisfy the constraint that ␪x,y = 0
i, t, h, m, then longest common subsequences (LCS) are                 when n10 + n01 = 0.
l, g, r, i, t, h, m and l, o, r, i, t, h, m, both of length 7. In   5. The maximum likelihood ᏸ(M) of marker order x1,
the weighted version of the problem, we look for com-                  x2, . . . , xm on a map M, also known as likelihood of
mon subsequence that has maximum weight. In the                        M, is
previous example, if the weights are a = 3 and for other
                                                                        ᏸ共M兲 = ᏸ共x1, x2, . . . , xm兲
letters 1, then the weighted common subsequence for
                                                                             = ᏸ共x1兲 × ᏸ共x2 | x1兲 × ᏸ关x3 | 共x1, x2兲兴 × . . .
A and B is a, r, i, t, h, m that has weight 8 and not l, g,
                                                                               × ᏸ关xm | 共x1, x2, . . . , xm1兲兴
r, i, t, h, m or l, o, r, i, t, h, m that have weight 7.
       The LCSP and its weighted version can both be
                                                                       We use ᏸ to denote multipoint likelihood and L to
solved using dynamic programming (Gusfield 1997).
                                                                       denote two-point likelihood. By considering condi-
The length of the LCS is often used to measure the
                                                                       tioning events as independent and removing the
similarity of two strings. We shall use it to quantify the
                                                                       conditioning on independent events, the multi-
consistency between a pair of maps where two or more
                                                                       point maximum likelihood of map x1, x2, . . . , xm is
markers are said to be consistent with a pair of maps if
                                                                       approximated by several two-point likelihood esti-
their partial order on both maps is the same; that is, if
                                                                       mates as
for every pair of markers x, y, either x < y in both maps,
x > y in both maps, or the relative positions of x, y are                   ᏸ共M兲 ≈ L共x1兲 × L共x2 | x1兲 × L共x3 | x2兲 × . . .
not specified in both maps.                                                        × L共xm | xm−1兲 = L共M兲                       (6)
Maximum Likelihood Computation                                      When there are many errors in the data, two-point
The steps for doing data analysis using maximum like-               likelihood estimates are preferred over multipoint like-
lihood are as follows (Boehnke et al. 1991; Lange et al.            lihood estimates because the errors should not propa-
1995):                                                              gate as badly. Our evaluation of rhvector data, as sum-


352     Genome Research
          www.genome.org


--- PAGE BREAK ---

                                                                                        Integrating Radiation Hybrid Maps


marized in Table 4, below, suggests that the error rate is                Computing retention frequency and breakage
high.                                                                probabilities for diploid data with errors results in
                                                                     Markov and hidden Markov models that can be used
TSP                                                                  for estimating the likelihoods by techniques such as
Given a finite number of cities and the cost of travel               the estimation-maximization (EM) algorithm. These
between each pair of them, find the cheapest way of                  methods are thus limited in the number of markers
visiting all of the cities and returning to the starting             they can map reliably and are not suitable for transla-
point. As explained in the Appendix, the TSP is intrac-              tion to TSP. Ben-Dor and Chor (1997) used the ap-
table in a formal sense, but much research has gone                  proach of first estimating the breakage probability be-
into methods for solving specific instances either ap-               tween every pair of markers, taking into account
proximately or optimally. A well-known software pack-                whether the data are haploid/diploid and contain labo-
age for the TSP, namely CONCORDE (Applegate et al.                   ratory errors instead of assuming that data are haploid
1998), has been shown to do fairly well even on huge                 and error free, and then reduced the MLE problem to
data sets and has set several world records for the larg-            TSP as above. They remark that using the breakage
est instances solved to optimality.                                  probability derived from the (degree five) polynomial
     In RH mapping, markers correspond to cities with                for diploid data did not always improve the results
a dummy marker as the start and end city, and the cost               compared with using the (degree two) polynomial for
of travel corresponds to the measures of similarity of               haploid data. Because the reduction for haploid error-
rhvectors. For haploid error-free data, the objective                free data can be used to approximate the likelihoods
functions for RH mapping can be translated into dis-                 for diploid data, we chose to compute the breakage
tance functions for TSP (Karp et al. 1996). We then                  probabilities assuming the data to be haploid and error
briefly state the reductions described in the reference.             free. We note that the ideas presented below can be
                                                                     extended to the case where breakage probabilities are
Reducing OCB to a Distance Measure for TSP                           derived from the polynomial for diploid data but the
The distance between two rhvectors (x1, x2, . . . , xn)              transformations to TSP are valid only for haploid error-
and (y1, y2, . . . , yn) is the number of positions at which         free data.
xi = 1 and yi = 0 or vice versa with distance from                        The reductions from OCB and MLE to TSP achieve
dummy marker to any other marker being any con-                      the corresponding objective function when the data
stant. If there are no unknowns in the given RH data,                does not have unknowns and is relatively error
then the marker order produced by TSP achieves mini-                 free. Recent advances in software for TSP, namely
mum OCB.                                                             CONCORDE, make it appealing to extend the above
                                                                     reductions to incorporate unknowns to reduce the ef-
Reducing MLE to a Distance Measure for TSP                           fect of unknowns on the quality of map produced us-
Define the transition probability for marker x as                    ing TSP. We then present five such extensions for the
                                                                     two reductions. Note that the reductions are a method
                     tx = 共公p兲n1共公q兲n0                        (7)
                                                                     for assigning edge weights in the TSP instance, not the
and the transition probability between markers x and                 method for evaluating the marker order on a map. The
y as                                                                 OCB and MLE objective functions are applied in the
                                                                     same way to a marker order, regardless of how the
     tx,y = 共1 − ␪x,yp兲n00共1 − ␪x,yq兲n11共␪x,y公pq兲n10+n01             marker order was obtained. To indicate which of the
                                                              (8)
                                                                     reductions from OCB or MLE to TSP is being extended,
tx is also referred to as the transition probability be-             we tag each name by TSP+OCB or TSP+MLE. We pre-
tween dummy marker and x. The transition probability                 sent them in terms of distances between a marker pair
of a map x1, x2, . . . , xm is given by                              x = (x1, x2, . . . , xn) and y = (y1, y2, . . . , yn). We use p
                                                                     and nij as before.
  T共x1, x2, . . . , xm兲 = tx1 × tx1,x2 × ⭈ ⭈ ⭈ × txm−1,xm × txm
                                                               (9)
                                                                     Normalized TSP+OCB
Karp et al. (1996) left it as an exercise to show that for
                                                                     The distance (n10 + n01) as computed in the reduction
haploid error-free data
                                                                     of OCB to TSP is normalized by n/(n00 + n01 + n10 + n11)
                 T共x1, x2, . . . , xm兲 = L共M兲                (10)    under the assumption that the positions with un-
                                                                     knowns in them have the same distribution of differ-
(See Appendix for a proof of equation 10.) The objec-
                                                                     ences as the positions in which both xi and yi are
tive in TSP is to minimize a sum of distances. To con-
                                                                     known. The distance according to this objective func-
vert the objective from maximizing a product to mini-
                                                                     tion is, then,
mizing a sum, suitable for TSP, set the distance dx,y as
ⳮlog(tx,y).                                                                     共n10 + n01兲 ⭈ n Ⲑ 共n00 + n01 + n10 + n11兲      (11)


                                                                                                     Genome Research           353
                                                                                                        www.genome.org


--- PAGE BREAK ---

Agarwala et al.


Weighted TSP+OCB                                                           −n ⭈ 共n1 ⭈ log 公p + n0 ⭈ log 公q兲
In this objective function, all six combinations for a                                   n0 + n1
pair from {0, 1, 2} are assigned a weight. We did several
experiments with different weighting schemes. Each             When the data does not have unknowns, the above
experiment has three steps: (1) compute edge weights           five extensions simplify to the two reductions men-
between every pair of markers, including the dummy             tioned earlier in this section.
marker, according to the weighting scheme, (2) solve                When OCB and MLE are incomparable, as in GB4
TSP by using the part of CONCORDE that guarantees              and G3 panel data, we should not expect solutions of
an optimal order for given distances (see Appendix for         TSP for each of the above five theoretically meaningful
details), and (3) compute OCB for the marker order M           and robust reductions to result in the same map. We
obtained by TSP; compute the sum of (n10 + n01) for            find the subset of markers for which order is not af-
consecutive markers on map M. Among the edge                   fected by the criteria used for placing them on a map.
weights that we tested, the scheme that results in a           Because each TSP+OCB [TSP+MLE] weighting scheme
map with lowest OCB is                                         is a minor variation of OCB [MLE] objective function,
                                                               we attribute the differences in marker order on maps to
n10 + n01 + 0.2 ⭈ n22 + 0.3 ⭈ 共n21 + n20 + n02 + n12兲   (12)   limitations of the data vectors and panels for the mark-
The schemes we tried were tested on the data we have           ers. The markers whose order is sensitive to the choice
for GB4 and G3 panels. As the above scheme gave                of reduction are removed in favor of constructing a
lower OCB and higher MLE for virtually all chromo-             reliable map at the cost of not placing every marker. In
somes and for both GB4 and G3 panel data, we believe           the next section we present how we can extract the
that the scheme should be generalizable to all human           pieces of the map that are consistent among all maps
radiation hybrid data. For example, consider the un-           to produce a single RH map for each panel and then
weighted scheme of (n10 + n01). The average number of          use the same idea to integrate the map for each panel.
breaks between consecutive markers for the marker or-
ders using the weighting scheme in equation 12 was
                                                               RESULTS
2.70 as against the unweighted scheme that had the
                                                               We first present a RH map construction strategy with
average of 2.79. The only case in which the un-
                                                               the goal of producing maps that can be integrated. The
weighted scheme did better was for GB4 panel data for
                                                               emphasis is on striking a balance between the reliabil-
chromosome 21 where the weighting scheme in equa-
                                                               ity of the map produced and the number of markers
tion 12 needed 2.36 average number of breaks and un-
                                                               that get placed on the map. Second, we present a map
weighted scheme needed 2.35.
                                                               integration strategy. The map integration procedure is
                                                               not specific to RH maps and can be used for any maps
Base TSP+MLE                                                   that have the same objective criteria. Third, we present
Same as reduction from MLE to TSP.                             comparisons of our new maps with previously pub-
                                                               lished maps, maps reconstructed with RHMAPPER, and
                                                               sequence data submitted to GenBank.
Extended TSP+MLE
Same as reduction from MLE to TSP except that in               Map Construction
equation 5, n is replaced by (n00 + n01 + n10 + n11).          The steps are as follows:

                                                               Step 1: Compute Framework Markers
Normalized TSP+MLE
                                                               The candidates C for framework markers are the mark-
The breakage probabilities are computed as in Ex-
                                                               ers typed on all panels. For each candidate framework
tended TSP+MLE. The transition probabilities are nor-
                                                               marker in C, its rhvectors from different panels are
malized to reflect that compution of breakage prob-
                                                               concatenated to produce a virtual rhvector for the
abilities ignores positions contributing to n22. The
                                                               marker. The set of framework markers F is a subset of
distance between x and y resulting from this normal-
                                                               framework candidate markers C such that no marker
ization is
                                                               pair in F is “very close” or “too ambiguous” to another
             −n关n00A + n11B + 共n10 + n01兲C +                   marker in F where closeness and ambiguity are deter-
       公共n + n 兲共B + C兲 + 共n + n 兲共A + C兲兴
              12    21               02   20
                                                               mined by cutoffs for break count B, negative logarithm
                                                               of transition probability LL, and percentage of un-
                         共n − n22兲                             knowns U. If a marker x僐C has more unknowns with
                                                               respect to the length of its rhvector than U, then x is
where A = log(1 ⳮ ␪x,yp), B = log(1 ⳮ ␪x,yq), and C =          not present in F. If a pair of markers x,y僐C have a break
log(␪x,y√pq). The distance between dummy marker and            count <B or have ⳮlog(tx,y)>LL, at least one of x,y is not
x is given by                                                  present in F. The breakage probability for tx,y was com-


354    Genome Research
         www.genome.org


--- PAGE BREAK ---

                                                                                      Integrating Radiation Hybrid Maps


puted as in Extended TSP+MLE. The cutoffs are deter-                  extendible piece is less (greater) than the interval to
mined experimentally and necessarily depend on the                    the last marker of the extendible piece, then the
data. We look for cutoffs that give a non-negligible set              piece is oriented from p-terminal to q-terminal (q-
F of framework markers such that the maps for markers                 terminal to p-terminal). If the first and last markers
in F computed in step 2 and step 3 are mostly consis-                 are assigned to the same interval, then the piece is
tent. For all maps described here, we used B = LL = 3                 unoriented.
but did not use any cutoff for percentage of unknowns.             4. If an extendible piece of M for fi can be oriented, the
                                                                      piece replaces fi, and relative ordering of markers in
Step 2: Compute Maps                                                  the piece is preserved; otherwise, all the markers in
Reduce the problem of computing a map to that of TSP                  the piece are collapsed at the position for fi.
using each of the five reductions described in the pre-
vious section. Use CONCORDE to solve each instance                 The global reordering of extendible pieces allows for
of TSP and transform the solution to a map. This re-               framework markers to be reordered locally on the map
sults in five maps for framework markers correspond-               when the extendible piece of M for fi contains the ex-
ing to five reductions.                                            tendible pieces of M for fi+1, fi+2, . . . , fi+k (resulting in
                                                                   empty extendible pieces of M for fi+1, fi+2, . . . , fi+k) and
Step 3: Compute a Framework Map
                                                                   the extendible piece of M for fi is oriented in the direc-
We compute a framework map as the map with only                    tion that puts fi+1 before fi. Thus, we are not treating
those framework markers whose order is consistent                  the framework map as absolutely rigid.
with all the maps computed in step 2. In practice, we                    We illustrate step 5 with the following example:
find that in step 1, deleting markers that have rhvec-             Let p-terminal, a3, a11, a8, a14, q-terminal be the frame-
tors with more unknowns than those conflicting with                work map computed in step 3 and let p-terminal, a1,
them is effective.                                                 a2, . . . , a14, q-terminal be a map computed in step 4.
                                                                   Suppose we assign the following: interval 0 to marker
Step 4: Compute Maps for Each Panel
                                                                   p-terminal, a1; interval 1 to markers a2, a3, a4; interval 2
Same as step 2 but with all markers for the panel and
                                                                   to marker a5; interval 5 to marker a6; interval 3 to
not just the framework markers.
                                                                   markers a7, a8, a9; interval 2 to markers a10, a11, a12;
Step 5: Reorder Maps                                               interval 4 to markers a13, a14; and interval 5 to marker
If there are m markers on the framework map, say f1,               q-terminal. Then, the extendible piece for p-terminal is
f2, . . ., fm, and two terminals (f0 for p-terminal and fm+1       p-terminal, a1, a2, a3, a4 ordered from p-terminal to a4;
for q-terminal), then there are m + 1 intervals on the             for a3 gets reduced from p-terminal, a1, a2, a3, a4, a5 to
framework map into which each remaining marker on                  just a5; for a11 is a7, a8, a9, a10, a11, a12 ordered from a12
the panel can be placed. For each marker x, we find the            to a7; for a8 gets reduced from a7, a8, a9, a10, a11, a12 to
interval fi, fi+1 such that the likelihood of fi, x, fi+1 is the   empty; for a14 is a13, a14 unordered; and for q-terminal
maximum among all the intervals. We compute the                    is q-terminal. Note that a6 does not get assigned to any
lod score of placing x as the logarithm of the likelihood          extendible piece. The map computed in step 4 gets re-
ratios of placing x in the best interval to placing x in           ordered in step 5 to p-terminal, a1, a2, a3, a4, a5, a12, a11,
the next best interval. Then, each map computed in                 a10, a9, a8, a7, a13, a14, q-terminal with a6 getting
step 4 is globally reordered as follows:                           dropped and a13, a14 collapsing to same position as
                                                                   that of framework marker a14.
1. For each fi, find the consecutive set of markers on                   The concatenation of rhvectors for the same
   map M including fi that have the interval i, i ⳮ 1 or           marker but different panels produces a longer virtual
   i + 1 assigned to them. This piece of the map is                rhvector. This gives us a better chance of obtaining a
   called an extendible piece of M for fi.                         reliable map for the common markers as we have more
2. Consider f0, . . . , fm+1 in order of their increasing          data to decipher their order on the map. If there is only
   index. For each fi, find the set of markers X in the            one panel for which we have to compute a RH map, we
   extendible piece for fi such that each marker in X is           do step 2 and step 3 described above using all markers
   also present in a previously considered extendible              available for the panel. However, when maps are to be
   piece for f0, f1, . . . , fiⳮ1. Delete markers in X from        constructed for more than one panel and these maps
   the extendible piece for fi. In practice, we do not see         are to be integrated, we devise our map construction
   markers in extendible pieces overlapping with                   strategy to take advantage of the fact that we have
   markers in extendible pieces of more than one or                some markers that are present in all panels. The reor-
   two previous framework markers.                                 dering of the map in step 5 results in some markers not
3. Determine if the assignment of interval i ⳮ 1 or                getting placed on the map. These markers are discarded
   i + 1 orients the piece with respect to the framework           because their vectors were not consistent with the
   map. If the interval assigned to the first marker of an         piece of the map that they were close to.


                                                                                                   Genome Research           355
                                                                                                      www.genome.org


--- PAGE BREAK ---

Agarwala et al.


Map Integration                                           RHdb/index.html). Before using the data from RHdb,
In this subsection we describe how to integrate two or    we first assign a unique identifier to each pair of for-
more maps that have the same criteria for measuring       ward and reverse primers for STS markers. Two markers
the score of placing a marker on a map. The core of the   with identical primer sequences are, in reality, the
integration strategy is to use the algorithm for the      same STS marker and are assigned the same identifier.
weighted LCSP for finding a set of markers that are       If an identifier has more than one rhvector, we pick an
common and have same relative order in a pair of          rhvector with the fewest unknowns.
maps.                                                          We reconstructed maps for the GB4 and G3 panel
                                                          data using CONCORDE and the five transformations
Merging Maps                                              to TSP (two variants of OCB and three variants of MLE)
To merge two maps, we first compute their weighted        described earlier. These maps were then integrated to
LCS. The markers common in both maps but not pres-        produce a single RH map. We have used the chained
ent in the LCS are deleted from both maps. The mark-      Lin–Kernighan (Lin and Kernighan 1973) heuristic
ers that are not common between the two maps are          from CONCORDE and the module that finds an opti-
interleaved by interpolation between markers that are     mal solution. Our experience is that the chained Lin–
in the LCS. For more than two maps, the number of         Kernighan heuristic from CONCORDE performs very
common markers among all of them may be consider-         well for RH data sets. For our data set, the running time
ably less than the number of common markers for any       for the number of iterations (250,000 kicks, two runs)
pair of maps. Our algorithm for merging more than         for which we ran chained Lin–Kernighan heuristic is
two maps is to first merge maps for all pairs and then    comparable with the running time of the module that
iteratively merge the results of those pairwise merged    finds an optimal solution. The module that finds an
maps. There is no fixed order in which pairwise maps      optimal solution requires a license for a software li-
are merged.                                               brary that is not free. To make the comparisons fair and
     For RH maps produced using the strategy in the       to make our software free to all, the results shown here
previous subsection, the weight of a marker is its lod    use only the free parts of CONCORDE.
score that is computed in step 5. The steps for produc-        The numbers of unique identifiers in the GB4
ing an integrated RH map use the merge procedure          panel and G3 panel data downloaded from RHdb are
described in the previous paragraph. The steps are as     40,898 and 7011, respectively. Each unique identifier
follows:                                                  corresponds to a marker in our analysis. The number of
                                                          markers common to both panels is 2087. Of these 2087
Step 6
                                                          markers, 1330 are candidates for the framework map as
Merge reordered maps to produce one map per panel.
                                                          the rest are too close to another candidate framework
Step 7                                                    marker. The number of markers placed on the frame-
Merge maps for each panel to produce an integrated        work map is 1084 with a maximum of 103 markers on
map.                                                      the framework for chromosome 4 and a minimum of
                                                          17 markers on the framework of chromosome 21. The
New Maps and Quality Assessment                           total number of markers placed on the integrated map
We have presented a method of constructing RH maps        is 23,723 out of 45,822 unique identifiers assigned to
from data on various panels with the aim of integrat-     the panel data.
ing them to produce a single RH map. We use our           Software Quality
algorithm on G3 panel and GB4 panel data down-            As described above, we have constructed an integrated
loaded from RHdb to construct an integrated G3/GB4        G3/GB4 RH map. We also attempted to produce maps
panel RH map. We seek to balance the quality of the       with RHMAPPER using the same data on both panels,
map and the number of markers that get placed on the      to compare RHMAPPER with our software that uses
map. Because the objective functions for RH mapping       CONCORDE. We chose RHMAPPER because it was
cannot be directly used to evaluate the quality of the    used to construct the Whitehead Institute map (Hud-
maps, we check our maps using segments of contigu-        son et al. 1995). We did not constrain RHMAPPER to
ous genomic sequence (contigs) reconstructed from in-     any initial framework or to any set of markers for the
dividual clone sequences (Jang et al. 1999), from chro-   initial framework. We computed an initial framework
mosome 22 sequence (Dunham et al. 1999), and with         using the options available in RHMAPPER on the panel
already published maps. We compare our software           data because we did not want to constrain RHMAPPER
with RHMAPPER.                                            to a possibly erroneous initial framework not of its own
                                                          choosing.
New Maps
We obtained the rhvector inputs for our experiments       Running Time
from the RH database (RHdb, http://www.ebi.ac.uk/         To compute all the single-chromosome maps described


356   Genome Research
        www.genome.org


--- PAGE BREAK ---

                                                                                   Integrating Radiation Hybrid Maps


above with CONCORDE took <2 weeks total on a Sun                 markers and the average of the logarithm of two-point
Ultra10 workstation. We even tested CONCORDE with                likelihood for the maps with breakage probabilities
input consisting of all the markers on all the chromo-           computed as in Extended TSP+MLE. We could not use
somes together, and that computation took 3 days. Be-            RHMAPPER for evaluating the multipoint likelihood of
cause we used the chained Lin–Kernighan heuristic                the maps because RHMAPPER suffers from underflow
from CONCORDE whose running time is dependent                    for the number of markers we have on our maps. We
on the number of iterations as well as the size of data,         compared maps computed using CONCORDE for both
computing a map of all the markers together with                 G3 and GB4 panel data and not our integrated map
(500,000 kicks, two runs) takes less time than the com-          because we cannot compute OCB or MLE when con-
putation of chromosome-specific map where each                   secutive markers on a map are from different panels.
map is run for (250,000 kicks, two runs). In contrast,           Furthermore, the maps produced by RHMAPPER are for
RHMAPPER could not finish a chromosome 1 map                     one panel. The results are summarized in Tables 1 and
within 3 weeks, and took >2 months to compute all the            2. RHMAPPER runs for chromosome 1 were aborted
remaining single-chromosome maps. Constraining to                after 3 weeks of computation and is reflected by a “?”
an initial framework would reduce the running time               in Tables 1 and 2. There, we have not attempted to
for RHMAPPER considerably but may impact the qual-               produce an “optimal” order for the markers that are
ity of the map and make the quality comparison done              binned to the same position on the maps because the
below invalid as the errors in maps computed by                  rhvectors for markers binned to the same position, in
RHMAPPER may be attributed to the initial framework.             principle, should be similar.
                                                                      Because the maps produced using RHMAPPER by
Map Comparison in Terms of OCB and MLE                           us (columns 2, 3, 5, and 6 of Table 1) have lower aver-
We consider Whitehead Institute map, maps com-                   age OCB and higher average logarithm of likelihood
puted by us using RHMAPPER for both G3 and GB4                   than the Whitehead Institute map, we feel that our
panel data, and maps computed using CONCORDE for                 strategy of not constraining RHMAPPER by an initial
both G3 and GB4 panel data. For each map and each                framework does not degrade the quality of maps pro-
chromosome, we compute the average number of                     duced. Therefore, it is fair to compare the maps pro-
chromosome breaks observed between consecutive                   duced by us using RHMAPPER with the maps produced



 Table 1. Average Number of Chromosome Breaks Between Consecutive Markers and the Average of the Logarithm of
 Two-Point Likelihood for the Maps with Breakage Probabilities Computed as in Extended TSP + MLE

                                 OCB/no. of markers                                 Log [L(M)]/no. of markers

 Chr           Whitehead             RHMAPPER         CONCORDE         Whitehead          RHMAPPER              CONCORDE

  1                3.70                    ?            1.66             ⳮ5.74               ?                   ⳮ2.28
  2                4.18                  3.80           2.12             ⳮ6.37              ⳮ5.47                ⳮ3.02
  3                3.92                  2.71           1.97             ⳮ6.17              ⳮ4.22                ⳮ2.82
  4                3.84                  3.75           2.15             ⳮ5.97              ⳮ5.37                ⳮ3.01
  5                3.66                  3.37           1.99             ⳮ5.72              ⳮ4.98                ⳮ2.73
  6                3.59                  2.60           1.70             ⳮ5.73              ⳮ4.07                ⳮ2.44
  7                4.00                  2.86           1.92             ⳮ6.20              ⳮ4.41                ⳮ2.79
  8                3.64                  3.64           2.09             ⳮ5.88              ⳮ5.39                ⳮ2.97
  9                3.56                  2.86           1.85             ⳮ5.59              ⳮ4.28                ⳮ2.55
 10                3.76                  3.55           2.04             ⳮ5.95              ⳮ5.24                ⳮ2.91
 11                3.35                  2.53           1.86             ⳮ5.26              ⳮ3.87                ⳮ2.42
 12                3.67                  3.87           1.98             ⳮ5.86              ⳮ5.66                ⳮ2.81
 13                3.58                  2.92           2.01             ⳮ5.72              ⳮ4.50                ⳮ2.89
 14                3.43                  2.43           1.79             ⳮ5.52              ⳮ3.78                ⳮ2.53
 15                4.28                  4.16           2.25             ⳮ6.64              ⳮ5.99                ⳮ3.19
 16                4.43                  3.18           2.32             ⳮ6.70              ⳮ4.87                ⳮ3.30
 17                4.27                  2.74           2.03             ⳮ6.71              ⳮ4.29                ⳮ2.83
 18                4.22                  3.07           2.47             ⳮ6.48              ⳮ4.76                ⳮ3.63
 19                4.40                  2.78           1.99             ⳮ6.74              ⳮ4.26                ⳮ2.67
 20                3.76                  2.41           1.74             ⳮ6.10              ⳮ3.83                ⳮ2.45
 21                3.99                  2.64           2.19             ⳮ6.65              ⳮ4.40                ⳮ3.35
 22                4.12                  2.87           2.17             ⳮ6.62              ⳮ4.53                ⳮ3.09
 X                 3.37                  2.36           1.70             ⳮ5.22              ⳮ3.61                ⳮ2.32

 (Chr) Chromosome number; (Whitehead) Whitehead Institute map (Hudson et al., 1995); (RHMAPPER) maps computed by us using
 RHMAPPER for GB4 panel data; (CONCORDE) maps computed using CONCORDE for GB4 panel data.



                                                                                               Genome Research        357
                                                                                                 www.genome.org


--- PAGE BREAK ---

Agarwala et al.


                                                           the initial framework map, RHMAPPER does local ex-
 Table 2. Average Number of Chromosome Breaks
 Between Consecutive Markers and the Average of the        tensions several times with random permutations of
 Logarithm of Two-Point Likelihood for the Maps with       the file containing information for triples. For growing
 Breakage Probabilities Computed as in Extended            the map by a marker, RHMAPPER considers only the
 TSP + MLE
                                                           triples created by consecutive markers on the initial
        OCB/no. of markers      Log [L(M)]/no. markers     framework map and the marker. There is an analogous
                                                           method for TSP called 2-opt (in general k-opt) that con-
 Chr RHMAPPER CONCORDE RHMAPPER CONCORDE
                                                           siders changing only two edges of the traveling sales-
  1        ?          2.91        ?           ⳮ4.41        man tour at a time and continues doing so until no
  2      4.96         2.88       ⳮ6.24        ⳮ4.40        further improvement can be found. It is established
  3      5.18         3.13       ⳮ6.55        ⳮ4.73        that for typical large TSP problems, the chained Lin–
  4      5.52         2.96       ⳮ6.77        ⳮ4.59
  5      5.17         3.09       ⳮ6.67        ⳮ4.69        Kernighan method in CONCORDE finds lower cost
  6      4.76         3.11       ⳮ6.28        ⳮ4.72        tours than 2-opt (e.g., see Johnson and McGeoch
  7      5.91         3.69       ⳮ7.16        ⳮ5.29        1997). This is because the Lin–Kernighan heuristic does
  8      5.34         3.09       ⳮ6.73        ⳮ4.70        more large-scale rearrangements and looks at a much
  9      4.73         3.16       ⳮ6.21        ⳮ4.79
 10      5.35         3.47       ⳮ6.73        ⳮ5.08        larger neighborhood of solutions than 2-opt to try to
 11      5.79         3.24       ⳮ7.00        ⳮ4.84        improve the current solution. We see no reason why
 12      5.31         3.32       ⳮ6.86        ⳮ5.05        this general difference in performance should be dif-
 13      4.58         3.17       ⳮ5.96        ⳮ4.79
 14      4.04         2.93       ⳮ5.69        ⳮ4.57
                                                           ferent for RH mapping problems. RHMAPPER does not
 15      4.70         3.76       ⳮ6.36        ⳮ5.39        formally treat the problem as an instance of TSP, but
 16      5.04         3.49       ⳮ6.47        ⳮ5.12        the heuristic used by RHMAPPER suffers from the same
 17      4.39         3.69       ⳮ6.07        ⳮ5.50        weakness that 2-opt has for TSP. It is an open research
 18      6.10         3.88       ⳮ7.52        ⳮ5.54
 19      4.95         3.23       ⳮ6.59        ⳮ4.88        problem to design and implement good software for
 20      4.87         3.70       ⳮ6.43        ⳮ5.53        finding best global order when information given is for
 21      3.79         3.36       ⳮ5.73        ⳮ5.16        only triples.
 22      4.21         3.41       ⳮ6.24        ⳮ5.20
 X       4.35         2.80       ⳮ5.52        ⳮ4.22
                                                           Quality of Integrated Map
 (Chr) chromosome number; (RHMAPPER) maps computed by      We compare the quality of our integrated map with
 us using RHMAPPER for G3 panel data; (CONCORDE) maps      that of previously published maps by looking at con-
 computed using CONCORDE for G3 panel data.                sistency with sequence data and by looking at the
                                                           maximum number of markers placed in same relative
                                                           order between pairs of maps.
using CONCORDE. Some of the differences in results
for GB4 panel between the map of Whitehead Institute       Consistency with Sequence Contigs
and the one produced by us using RHMAPPER on GB4           To test the correctness of a map, we can check the
panel can be attributed to the fact that we are using      order of markers that are on the contigs that have been
markers that became available since their map was          sequenced. We place markers on contigs using the e-
published. It is also possible that the currently avail-   PCR program (Schuler 1997). On October 27, 1999,
able data have been cleaned since earlier versions that    there were 1807 human DNA contigs in GenBank on
may have been used for previous maps or that the ge-       which we placed at least one marker. The position of a
netic map and YAC contig data used by the Whitehead        marker on a contig was determined by the physical
Institute to build the initial framework was erroneous.    base pair position of the left end of the marker from
     CONCORDE consistently produces maps that              one end of the contig. The number of pairs of markers
have lower average OCB and higher average logarithm        that were consecutive on a contig and typed on GB4
of likelihood than those constructed with RHMAPPER.        and/or G3 panels were 4071 and 98, respectively. We
Because computation of maps using RHMAPPER                 say that a contig is consistent with the map if there are
and our software started with the same data and            at least three markers that are both on the map and on
RHMAPPER was not constrained by an initial (possibly       the contig under consideration and the order of these
erroneous) framework map, Tables 1 and 2 suggest that      markers is the same. Our analysis considers all the
our strategy is able to do a better job than RHMAPPER.     markers on the map and is not restricted to the markers
Based on work for TSP, there is some intuitive justifi-    that are placed with significant statistical support. We
cation for why this should be so. First, orders of mag-    also consider the case when one marker is allowed to
nitude more person years have been spent developing        be misplaced on the contig. The number of consistent
algorithms and software for TSP than for RH mapping.       contigs are 159 of 799 (19.90%) for GB4 map of RH
Second, the approach taken by RHMAPPER is to con-          Consortium (Deloukas et al. 1998), 97 of 291 (33.33%)
sider triples of markers and to do local extensions. For   for GB4 map of Whitehead (Hudson et al. 1995), 27 of


358    Genome Research
         www.genome.org


--- PAGE BREAK ---

                                                                                        Integrating Radiation Hybrid Maps


84 (32.14%) for G3 map of Stanford (Stewart et al.                sequence suggest that marker 55194 is contained in
1997), and 199 of 496 (40.12%) for the integrated map             marker 77310, which is clearly disputed by the rhvec-
we produced. The number of contigs that become con-               tors for two markers as they differ in 41 positions. For
sistent when one marker is allowed to be misplaced are            such extreme discrepancies, the RH mapping strategy
318 of 799 (39.80%) for GB4 map of RH Consortium                  is clearly not the cause of the inconsistency, and there
(Deloukas et al. 1998), 162 of 291 (55.67%) for GB4               is some experimental error. We believe, and analysis of
map of Whitehead (Hudson et al. 1995), 46 of 84                   Ben-Dor and Chor (1997) suggests, that smaller dis-
(54.76%) for G3 map of Stanford (Stewart et al. 1997),            crepancies like those in Table 4, below, are unavoidable
and 309 of 496 (62.30%) for the integrated map we                 and affect the map computation because of the small
produced. By each measure, our map is better than the             size of RH panels currently in use. We cannot rule out
other three maps. The number of contigs that could be             some other types of errors in conducting the RH ex-
considered is lower for the integrated map than for the           periments or in depositing the data in RHdb, but the
RH Consortium map because the number of markers                   error rates would have to be extremely high to account
on the integrated map is lower. Furthermore, the num-             for the inconsistencies between rhvectors and contigs.
ber of consistent contigs is higher, which can be
viewed as evidence that we are not deleting too many              Consistency with Chromosome 22 Sequence
markers and we are deleting markers with problematic              The completed sequence for chromosome 22 is avail-
data. However, the contig data and maps produced are              able from http://www.sanger.ac.uk/HGP/Chr22/ (Dun-
still very inconsistent. Inconsistencies can arise either         ham et al. 1999). It consists of 12 contiguous segments
because (1) the contig data have many errors, (2) the             covering 33.4 million bp separated by 11 gaps of
mapping procedure is incorrect, or (3) the RH data                known size. The availability of chromosome 22 se-
have many errors. Evidence that either the contig or              quence allows us to consider only the markers that are
RH data are incorrect, and not the mapping strategy,              placed reliably on a chromosome 22 map and to find
comes from looking at the rhvectors of the markers                out the percentage of these markers that are in the
that are placed consecutively on a contig. We check               same order as the chromosome 22 sequence. Table 3
whether the contig data are consistent with the RH                summarizes the results for the RH Consortium, White-
data by looking at the OCB distance (rhvector differ-             head, Stanford, and our integrated maps. It is shown
ences) between rhvectors for the markers consecutively            that the integrated map consistently does better than
placed on a contig. Table 4, below, summarizes the                the RH Consortium and Whitehead maps. In places
OCB distance observed between markers that were                   where the integrated map does not do as well in per-
placed consecutively on a contig. For a RH mapping                cent of markers correct as the Stanford map, we are
strategy to place markers consecutively, the rhvectors            considering almost three times as many markers as the
of these markers should be close to each other. There-            Stanford map. The Généthon map could not be con-
fore, no plausible RH mapping strategy should place               sidered for Table 3 as it does not assign reliability to
markers consecutively if they have more than two or               placement of markers.
three differences. We found many cases where two
markers that are consecutive on sequence contigs have             Map Comparison in Terms of LCS
many differences in their rhvectors. Consider, for ex-            We consider every pair of Généthon, RH Consortium,
ample, markers on GenBank entry AC004231 shown in                 Stanford, and our integrated map. Table 6 lists the
Table 5, below. The physical base pair positions on the           number of markers that are common between a pair of



           Table 3. Number of Markers that Are in Same Order on the Map as Chromosome 22 Sequence Out
           of the Top K% of Markers on the Map

           Top K%             RH Consortium              Integrated               Whitehead                   Stanford

             5                   14/23 (61)               11/12 (92)                8/10 (80)                  4/4 (100)
            10                   29/46 (63)               19/25 (76)               15/20 (75)                  6/8 (75)
            15                   45/70 (64)               28/38 (74)               20/30 (67)                10/13 (77)
            20                   57/93 (61)               37/51 (73)               27/40 (68)                12/17 (71)
            25                  74/117 (63)               47/64 (73)               33/50 (66)                16/22 (73)
            50                 107/234 (46)              84/129 (65)              60/100 (60)                30/44 (68)
            75                 145/351 (41)             111/194 (57)              78/150 (52)                49/66 (74)
           100                 183/469 (39)             151/259 (58)             100/201 (50)                66/89 (74)

           The markers are sorted by lod score and the top-most marker has the best lod score. Percentages are given
           in parentheses.



                                                                                                        Genome Research      359
                                                                                                            www.genome.org


--- PAGE BREAK ---

Agarwala et al.


                                                              from Généthon’s genetic map for constructing their
 Table 4. Number of Markers Pairs that Are
 Consecutive on a Contig But Have Rhvectors at                initial framework map.
 OCB Distance

 OCB                GB4 rhvectors            G3 rhvectors
                                                              DISCUSSION
                                                              We presented a method for producing RH maps that
 0                     595 (15)                  0 (0)        robustly treats the data currently available. Some steps
 1                     764 (19)                 33 (34)       in the process can be further optimized. In particular,
 2                     674 (17)                 19 (19)
 3                     567 (14)                 27 (28)
                                                              one would like to have a mechanism in which vectors
 4                     409 (10)                 10 (10)       with errors can be detected before the TSP is used to
 5                     311 (8)                   5 (5)        construct a map. This would decrease the number of
 6                     209 (5)                   1 (1)        markers that are thrown out in step 5 of our algorithm.
 7                     162 (4)                   1 (1)
 8                     123 (3)                   1 (1)             We demonstrated with markers from the two larg-
 9                      74 (2)                   0 (0)        est human RH maps currently available (Stewart et al.
 10                     58 (1)                   0 (0)        1997; Deloukas et al. 1998) that our map integration
 11                     34 (1)                   1 (1)
                                                              strategy produces maps that are more consistent with
 12                     28 (1)                   0 (0)
 13                     25 (1)                   0 (0)        sequence data in GenBank than either map alone. This
 14                     13 (0)                   0 (0)        validates the hypothesis that integrated maps can add
 15–24                  18 (0)                   0 (0)        value over nonintegrated maps. However, our inte-
 25–34                   4 (0)                   0 (0)
 35–44                   3 (0)                   0 (0)        grated map is still quite inconsistent with sequence
 >44                     0 (0)                   0 (0)        data, and we showed that this is largely due to poor
                                                              data quality that cannot be easily overcome by better
 Percentages are in parentheses.                              mapping algorithms. The inconsistency between
                                                              rhvectors and DNA sequence contigs casts doubt on
                                                              the hypothesis that adding more markers to current
maps and the number of markers in their LCS. A LCS            RH maps can guide future DNA sequencing effectively.
between a pair of maps gives the largest subset of mark-           To compute our integrated chromosome maps, we
ers whose relative order on both maps is the same. As         found it necessary to first recompute maps based on
expected, the number of markers common between                the previously used markers, so as to take advantage of
the integrated map and G3/GB4 is more than the num-           some markers that were typed on multiple panels. Re-
ber of markers common between G3 and GB4 maps.                computing the initial maps was practical only because
The integrated map looks more consistent with the G3          the genomics research community has had the fore-
map than with the GB4 map, when consistency is mea-           sight to insist on making sequence, marker, and rhvec-
sured in terms of the length of the LCS of markers. It is     tor data freely available. The recomputation process
interesting to note that 82.42% of markers are in LCS         confirmed serious concerns raised by Ben-Dor and
between our integrated map and Généthon’s genetic           Chor (1997) about how RH mapping is being per-
map that was not constrained by any initial framework         formed in practice.
map as against 95.76% and 90.49% of markers for GB4                Ben-Dor and Chor (1997) presented both theoreti-
and G3 maps, respectively, which used information             cal and practical assessments of RH mapping methods.
                                                              On the practical side they tested the usage of TSP to
                                                              construct maps. They suggested that computing RH
 Table 5.     Data for Markers on Sequence AC004231
                                                              maps via the reduction to TSP could produce maps of
                                                Distance      comparable quality to RHMAPPER. However, they used
 Marker        Radiation                                      smaller data sets than ours, and used only three simple
 identifier   hybrid name           bp         bp     OCB
                                                              heuristics for TSP. We pushed their suggestion much
 61868          RH55030     124404..124553       —     —      further by using much larger data sets and using the
 72154          RH39412     127258..127493     2854     4     CONCORDE software package for TSP. The results both
 77310          RH13349     149852..150089    22594    42     in terms of map quality (Tables 1 and 2), and running
 55194          RH55082     149856..150027        4    41
 52532          RH18130     173790..174030    23934     2     time are striking. CONCORDE consistently produces
 19032          RH55096     173894..174232      104     0     maps that have fewer OCB and higher maximum like-
 64513          RH46938     188674..188820    14780     0     lihood than published maps and maps recomputed
 62207          RH28210     211868..212062    23194     0
                                                              with RHMAPPER. Moreover, CONCORDE could easily
 80183          RH47583     214891..215050     3023    10
 21845          RH55137     215148..215274      257     4     handle all the data for each chromosome, computed all
 12533          RH46475     220582..220701     5434     3     our single chromosome maps in under 2 weeks, and
                                                              was even able to compute a map using all markers
 Base pair difference is taken from left end. Distances are   from all chromosomes together in 3 days. In contrast,
 shown with respect to the previous marker.
                                                              RHMAPPER without a precomputed initial framework


360      Genome Research
          www.genome.org


--- PAGE BREAK ---

                                                                                      Integrating Radiation Hybrid Maps



 Table 6. Number of Markers that Are in the Same Order Between a Pair of Maps Out of the Number of Markers that Are
 Common Between Them

 Chr            Gnt vs. Int           Gnt vs. GB4      Gnt vs. G3          G3 vs. GB4          Int. vs. GB4        Int vs. G3

  1             106/141                 121/133          53/61               150/199           1149/2074             338/428
  2              65/83                   55/58           64/75                98/136             817/1223            287/355
  3              90/106                  65/66           68/81                96/132             815/1172            324/410
  4              43/52                   67/68           22/22                74/99              549/877             444/517
  5              41/47                   34/34           24/26                46/53              733/900             181/220
  6              79/98                  123/127          29/33                72/89              825/1268            260/295
  7              62/78                   49/49           53/63                78/102             530/803             236/295
  8              32/37                   27/27           24/24                56/65              421/734             193/227
  9              34/38                   24/25           27/28                57/77              444/640             161/210
 10              57/65                   54/59           33/37                73/96              649/925             219/268
 11              49/58                   41/41           42/45                79/102             788/1037            275/305
 12              51/57                   57/60           32/33                49/67              680/1038            213/236
 13              26/33                   35/35           18/19                33/34              247/436             119/141
 14              29/35                   30/30           23/23                54/66              463/707             185/211
 15              22/31                   29/30           16/17                37/53              378/639             108/159
 16              34/40                   34/35           28/28                32/38              370/519             137/150
 17              25/29                   26/26           14/14                46/62              487/782             139/153
 18              26/28                   26/27           16/16                31/43              184/326             109/140
 19              19/24                   25/25           12/12                25/43              381/591             136/147
 20              46/55                   44/50           19/20                41/46              378/582             123/133
 21              17/22                   17/18           11/13                16/20              156/228              99/113
 22               9/12                   15/17            7/9                 16/20              147/251              78/82
 23              32/37                   42/46           12/16                22/28              383/528              68/112
   Total        994/1206              1040/1086         647/715            1281/1670          11974/18280          4432/5307
                (82.42%)               (95.76%)        (90.49%)             (76.71%)            (65.50%)            (83.51%)

 (Chr) Chromosome number; (Gnt) Généthon map (Dib et al. 1996); (Int) our integrated map; (GB4) RH Consortium map (Deloukas
 et al. 1998); (G3) Stanford map (Stewart et al. 1997).



did not finish the chromosome 1 map within 3 weeks.                 minimum number of markers on our framework maps
Thus, our map construction strategy is the first one                are 103 for chromosome 4 and 17 for chromosome 21,
than can be scaled up to handle many more markers                   respectively. Using the larger m = 93 (GB4 panel) and
than are currently available without being pinned to a              103 and 17 markers, the chance of ordering 103 and 17
possibly erroneous framework. We show that RH map-                  randomly chosen markers correctly is <3.6% and 58%,
ping can be done efficiently by taking advantage of the             respectively. Although the theorem as stated does not
theoretical work and software package developed for                 apply when one selects a subset of markers, which may
solving a general combinatorial optimization problem.               be easier to order correctly, it does suggest that sticking
     On the theoretical side, Ben-Dor and Chor (1997)               to a rigid framework is unlikely to work well. String-
raised serious doubts about the ability of the RH ap-               ham et al. (1999) propose a way of not relying com-
proach to produce good maps with current panel sizes.               pletely on a fixed framework map but do not produce
We rewrite theorem 3 of Ben-Dor and Chor (1997) as                  a whole genome map based on that method. To avoid
follows:                                                            the Ben-Dor and Chor lower bound, one should
                                                                    choose the framework markers carefully and allow for
Theorem                                                             the possibility of rearranging or changing the frame-
The success probability s of correctly ordering n uni-              work markers in light of the other data. Our method is
formly distributed markers is bounded by                            not pinned to a framework map and allows for the
                                  1                                 possibility of framework markers to be rearranged lo-
                  sⱕ                                   (13)         cally in step 5 followed by possible removal of frame-
                       1 + 关n Ⲑ 共2mpq␭兲兴
                              2
                                                                    work markers during merge in step 6 and step 7.
where m is the number of hybrids, p is the retention                     Moreover, from inequality 13, it follows that panel
probability, q = 1 ⳮ p, and ␭ is the intensity of the               sizes must be 2 orders of magnitude larger than cur-
breakage process.                                                   rently used to boost the success probability signifi-
    Using the parameter values of n = 200, m = 83 (for              cantly. It is not the case that using too small panel sizes
G3 panel), p = 0.3, ␭ = 10 (Ben-Dor and Chor 1997),                 simply causes local rearrangements that can be ignored
the above inequality shows one has a <1% chance of                  by the commonly used practice of binning nearby
finding the correct marker order. The maximum and                   markers. Rather, we find that marker pairs that belong


                                                                                                   Genome Research        361
                                                                                                     www.genome.org


--- PAGE BREAK ---

Agarwala et al.


close together according to the DNA sequence often                 hereby marked “advertisement” in accordance with 18 USC
have a very large number of OCB in their rhvectors.                section 1734 solely to indicate this fact.
This indicates that either the data quality is poor or the
panel sizes are too small, so that the essential assump-
                                                                   APPENDIX
tion that nearby markers have nearby rhvectors does
not hold with high enough probability. Furthermore,                TSP
at present there is no good way to assess the fraction of          If a salesman, starting from his home city, is to visit
markers correctly ordered on a map. This confirms the              exactly once each city on a given list and then return
theoretical evidence by Ben-Dor and Chor (1997) that               home, he could select the order in which cities are
adding more markers without increasing the panel size              visited in such a way that the total distances traveled is
is not a fruitful strategy to obtain maps with better              as small as possible. Even when he knows the distance
quality.                                                           between every pair of cities, it is not at all clear how the
     In sum, there is a map integration and reconstruc-            data should be used to get the tour of minimum dis-
tion strategy that can produce maps with better qual-              tance efficiently. This is called the TSP.
ity. Our software improves current technology for do-                   Mathematically, an instance of TSP is composed of
ing the RH mapping in areas of computation time and                a number n of cities and an n ⳯ n distance matrix
algorithms for considering large number of markers for             D = [dij], where each dij is a non-negative integer and
mapping. The essential impediments to producing                    the question asked by the TSP is, “What is the shortest
dense high-quality RH maps are data quality and panel              tour for the n cities?” It is well known that TSP belongs
size, not computation.                                             to a class of problems called NP-Complete problems
                                                                   (Garey and Johnson 1979). Loosely speaking, this is a
                                                                   class of problems for which there is no known polyno-
METHODS                                                            mial time algorithm, and also for any pair of problems
The maps are stored in SQL Server Release 11.0.x of the Sybase     belonging to this class, one can be reduced to the other
database management software. The functions are imple-
                                                                   in polynomial time. So, if any problem that belongs to
mented using Transact-SQL and C version of Open Client
DB-Library. The algorithm was developed on Unix System V           this class can be solved in polynomial time, all of them
release 4.0 running under SunOS 5.5.1 using Sun WorkShop           will become solvable in polynomial time. This suggests
Compiler C 4.2, but it is compatible with other Unix comput-       that a fast algorithm for the TSP is unlikely to exist.
ers. The mapping software and a copy of this paper are avail-           Work in complexity theory (Karp 1972) indicates
able via an electronic mail request to richa@helix.nih.gov.        that problems like TSP are probably inherently expo-
The integrated G3/GB4 marker map is available at http://
                                                                   nential, that is, the computing time grows exponen-
www.ncbi.nlm.nih.gov/genome/rhmap. For each chromo-
some, the recomputed integrated map has columns for (1)
                                                                   tially with the number of cities. In view of the compu-
marker name, (2) position (cR) on recomputed GB4 map, (3)          tational difficulties in obtaining optimal tours, a num-
odds on recomputed GB4 map, (4) position (cR) on recom-            ber of algorithms have been developed that run faster
puted G3 map, and (5) odds on recomputed G3 map. In our            but do not necessarily produce an optimal tour (Lawler
computation, we assigned a unique identifier to each marker.       et al. 1985; Reinelt 1994).
For each marker its unique identifier and the following infor-          However, even though the TSP is hard in general,
mation, when available, can be obtained by clicking on the
                                                                   in practice the situation is not hopeless. The software
marker name: (1) primer information, (2) aliases for marker
name, (3) mapping information with respect to GeneMap’99,          package CONCORDE (Applegate et al. 1998) provides
and (4) e-PCR results on genomic contigs and cDNAs as ap-          two primary tools for solving TSPs. The first is a
propriate. Information on CONCORDE is available at http://         chained Lin–Kernighan heuristic (Lin and Kernighan
www.caam.rice.edu/keck/concorde.html. Finished sequences           1973; Martin et al. 1991). Any two cities are connected
of individual clones produced by the Human Genome Project          by an edge whose cost is the distance between the two
have been merged into contiguous sequence segments (con-           cities. The Lin–Kernighan heuristic is a local improve-
tigs) as previously described (Jang et al. 1999). The positions
                                                                   ment heuristic that starts with an initial tour (e.g., a
of markers within these sequences were determined by the
e-PCR program (Schuler 1997), using a word size of six             “nearest-neighbor” tour that at each step goes to the
(W = 6), a variability of up to 10 bases in the PCR product size   nearest city not already in the tour) and then repeat-
(M = 10), and up to 1 mismatching base allowed (N = 1).            edly searches for a set of edges in the current tour that
                                                                   can be exchanged with a set of edges not in the current
                                                                   tour, shortening the length of the tour. Lin–Kernighan
ACKNOWLEDGMENTS                                                    generalizes the 2-Opt and 3-Opt heuristics, which only
We thank David Lipman and James Ostell for helpful discus-         consider exchanges of sets of edges of size 2 and 3,
sions. We are indebted to the reviewers of this manuscript for
carefully reading it and making several extremely useful sug-
                                                                   respectively. Chained Lin–Kernighan uses the Lin–
gestions that have improved the exposition of our work.            Kernighan heuristic, but when Lin–Kernighan fails to
     The publication costs of this article were defrayed in part   find an improving exchange, it repeatedly applies a
by payment of page charges. This article must therefore be         random “kick” (a four-edge exchange that is not easily


362     Genome Research
          www.genome.org


--- PAGE BREAK ---

                                                                                   Integrating Radiation Hybrid Maps


made by Lin–Kernighan) to the tour, reruns the Lin–         ies and has obtained tours provably within 0.11% of
Kernighan heuristic, and keeps the new tour if the kick     optimal for the five remaining problems (with 14,051–
plus Lin–Kernighan resulted in an improvement. Be-          85,900 cities). On a modern workstation, the chained
cause the kick is a relatively small disruption to the      Lin–Kernighan heuristic obtains a tour provably with
tour, the Lin–Kernighan process after a kick is much        1.0% of optimal within 1 min for every TSP problem in
faster than the first Lin–Kernighan process from the        TSPLIB.
initial tour.
     The second tool provides a lower bound on all
tours. This lower bound is used to prove that a tour is
                                                            Equivalence of Likelihood and Transition
optimal or to obtain a quality guarantee for a tour. The    Probabilities for Haploid Error-Free Data
approach CONCORDE uses to establish a lower bound,          For haploid error-free data, equation 1 gives
introduced by Dantzig et al. (1954), is to consider a                                 L共x兲 = pn1 × qn0                      (14)
linear relaxation of the TSP. Linear relaxation means
that the problem is reformulated as minimizing a lin-       and equation 2 gives
ear objective function subject to a set of linear in-
                                                                        L共x, y兲 = 关p共1 − ␪x,yq兲兴n11 × 关pq␪x,y兴共n01+n10兲
equalities. The linear relaxation is different from the
TSP because in TSP some variables must have integer                               × 关q共1 − ␪x,yp兲兴n00                       (15)
values, but the linear relaxation drops the constraint      We prove equation 10 by the induction on length of
that variables must take on integer values.                 map.
     The linear relaxation is solved using the simplex
method (Papadimitriou and Steiglitz 1982). To work          Base Case
back from the solution of the linear relaxation to a        (m = 1 ). From equation 1
solution for the TSP instance, the solution is refined by
                                                                              L共x兲 = pn1 × qn0 = tx × tx = T共x兲
adding “cutting planes,” linear inequalities that are
true for every tour, but violated by the solution of the    Induction Hypothesis
current linear relaxation. Because at each step we are      Suppose the claim holds for all maps of length k < m.
considering a relaxation of the TSP, the solution of the
relaxation provides a lower bound on the solution of        Induction Step
the TSP. If CONCORDE is unable to find any cutting          Assume k = m. From equation 6, we want to show that
planes for the current solution or if the lower bound
                                                                  T共x1, x2, . . . , xm兲 = L共x1兲 × L共x2 | x1兲
has ceased improving over a series of cutting planes
                                                                                          × L共x3 | x2兲 . . . L共xm | xm−1兲
CONCORDE resorts to branching. Because for any
nonempty proper subset of the cities a tour must enter      Substituting for T(x1, x2, . . . , xm) from equation 9 and
or leave the subset a positive even number of times,        using induction hypothesis, we get
CONCORDE branches by selecting a nonempty proper
                                                              T共x1, x2, . . . , xm兲 = tx1 × tx1,x2 × ⭈ ⭈ ⭈ × txm−1,xm × txm
subset of the cities, splitting the problem into two sub-
problems, one in which the solution is permitted to                                 = T共x1, x2, . . . , xm−1兲 × txm−1,xm
enter or leave the subset only twice and the other in                                 × txm Ⲑ txm−1
which the solution is required to enter or leave the                                = L共x1兲 × L共x2 | x1兲
subset at least four times. CONCORDE then recursively                                 × L共x3 | x2兲 . . . L共xm1 | xm−2兲
applies the same procedure to each subproblem. Once                                   × txm−1,xm × txm Ⲑ txm−1
branching has begun, the weakest lower bound from
the subproblems provides a lower bound for the TSP.         Therefore, to prove the claim, it is sufficient to prove
Of course, if the lower bound in any subproblem ex-         that
ceeds the length of the best known tour, that subprob-               共txm−1,xm × txm Ⲑ txm−1兲 = L共xm | xm−1兲
lem can be pruned. As a result, when solving a TSP, the                                       = 关L共xm−1, xm兲L共xm−1兲兴
chained Lin–Kernighan heuristic is applied to obtain a
very good tour prior to branching, so that subproblems      Rewriting the above equation and using notation x =
may be pruned more readily.                                 xm-1, y = xm, and nji is the number of times i occurs in
     These two tools are very effective at handling even    rhvector for marker j, it is sufficient to prove that
moderately large TSPs. TSPLIB (Reinelt 1991), available                                       tx,y × ty × L共x兲
at http://www.iwr.uni-heidelberg.de/iwr/comopt/soft/                               L共x,y兲 =
                                                                                                      tx
TSPLIB95/TSPLIB.html, is a library of TSP and related
variants that provides a benchmark of the state of the        We first use equation 15 after substituting nx1 = (n10 +
art in solving TSPs. CONCORDE has been used to solve        n11), nx0 = (n01 + n00), ny1 = (n01 + n11), and ny0 = (n10 +
every TSP problem from TSPLIB with up to 13,509 cit-        n00).


                                                                                                  Genome Research           363
                                                                                                     www.genome.org


--- PAGE BREAK ---

Agarwala et al.


 L(x, y) = (from equation 15兲                                                  Spillett, D. Muselet, J.-F. Prud’Homme, C. Dib, C. Auffray, J.

            关(1 − ␪x,yp) (1 − ␪x,yq) 共␪x,y公pq兲       兴
                                                                               Morissette, J. Weissenbach, and P.N. Goodfellow. 1996. A
                           n00                    n11           n10+n01
                                                                               radiation hybrid map of the human genome. Hum. Mol. Genet.

            × 共 公p 公q 兲 × 共 公p 公q 兲
                   n
                       y
                          n
                                 y
                                      n    n
                                                   x        x                  5: 339–346.
                       1         0                 1        0
                                                                           Heath, S.C. 1997. Markov chain monte carlo methods for radiation

            关(1 − ␪x,yp)n (1 − ␪x,yq)n (␪x,y公pq兲n +n 兴
                            00                     11            10   01       hybrid mapping. J. Comp. Biol. 4: 505–517.
                                                                           Hudson, T.J., L.D. Stein, S.S. Gerety, J. Ma, A.B. Castle, J. Silva, D.K.
                  × 共 公pn 公qn 兲 × 共 公pn 公qn 兲
                                 y        y                 x   x
                                 1        0                 1   0              Slonim, R. Baptista, L. Kruglyak, S.-H. Xu et al. 1995. An
          =                                                                    STS-based map of the human genome. Science 270: 1945–1954.

                            共公pn 公qn 兲
                                              x         x
                                              1         0                  Jang, W., H.C. Chen, H. Sicotte, and G.D. Schuler. 1999. Making
                                                                               effective use of human genomic sequence data. Trends Genet.
                                                                               15: 284–286.
          = (from equation 14)                                             Johnson, D.S. and L.A. McGeoch. 1997. The traveling salesman

            关(1 − ␪x,yp)n (1 − ␪x,yq)n 共␪x,y公pq兲n +n 兴
                                                                               problem: A case study in local optimization. In Local search in
                            00                     11            10   01
                                                                               combinatorial optimization (eds. E.H.L. Aarts and J.K. Lenstra), pp.

                         × 共 公pn 公qn 兲 × L(x兲
                                          y         y                          215–310. John Wiley and Sons, London, UK.
                                          1         0
                                                                           Karp, R.M. 1972. Reducibility among combinatorial problems. In

                             共公pn 公qn 兲
                                              x         x                      Complexity of computer computations (eds. R.E. Miller and J.W.
                                              1         0
                                                                               Thatcher), pp. 85–103. Plenum Press, New York, NY.
                                                                           Karp, R.M., W.L. Ruzzo, and M. Tompa. 1996. “Algorithms in
                                                                               molecular biology—Lecture notes,” Department of Computer
          = (from equation 8)
                                                                               Science and Engineering, University of Washington, Seattle, WA.
                     共公p 公q 兲 × L(x)
                            y         y
            tx,y ×         n1        n0                                    Lange, K., M. Boehnke, D.R. Cox, and K.L. Lunetta. 1995. Statistical
                                                                               methods for polyploid radiation hybrid mapping. Genome Res.
                     共公p 公q 兲
                            x         x
                           n1        n0                                        5: 136–150.
                                                                           Lawler, E.L., J.K. Lenstra, A.H.G. Rinnooy Kan, and D.B. Shmoys.
                                                                               1985. The traveling salesman problem: A guided tour of
          = (from equation 7)                                                  combinatorial optimization. Wiley-Interscience Series in Discrete
            tx,y × ty × L(x)                                                   Mathematics. John Wiley & Sons, New York, NY.
                                                                           Lin, S. and B.W. Kernighan. 1973. An effective heuristic for the
                    tx                                                         traveling salesman problem. Operations Res. 21: 498–516.
                                                                           Lunetta, K.L., M. Boehnke, L. Lange, and D.R. Cox. 1996. Selected
REFERENCES                                                                     locus and multiple panel models for radiation hybrid mapping.
Applegate, D., R. Bixby, V. Chvátal, and W. Cook. 1998. On the                Am. J. Hum. Genet. 59: 717–725.
   solution of traveling salesman problems. Documenta Math.III,            Martin, O., S.W. Otto, and E.W. Felten. 1991. Large-step Markov
   (http://www.mathematik.uni-bielefeld.de/documenta/Welcome-                  chains for the traveling salesman problem. Complex Syst.
   eng.html) International Congress of Mathematics III: 645–656.               5: 299–326.
Ben-Dor, A. and B. Chor. 1997. On constructing radiation hybrid            Matise, T.C. and A. Chakravarti. 1995. Automated construction of
   maps. J. Comp. Biol. 4: 517–533.                                            radiation hybrid maps using MultiMap. Am. J. Hum. Genet.
Boehnke, M., K. Lange, and D.R. Cox. 1991. Statistical methods for             57: A15, meeting abstract.
   multipoint radiation hybrid mapping. Am. J. Hum. Genet.                 Matise, T.C., J.J. Wasmuth, R.M. Myers, and J.D. McPherson. 1998.
   49: 1174–1188.                                                              Somatic cell genetics and radiation hybrid mapping. In Genome
Cox, D.R., M. Burmeister, E.R. Price, S. Kim, and R.M. Myers. 1990.            analysis: A laboratory manual, pp. 259–302. Cold Spring Harbor
   Radiation hybrid mapping: A somatic cell genetic method for                 Laboratory Press, Cold Spring Harbor, NY.
   constructing high-resolution maps of mammalian chromosomes.             Papadimitriou, C.H. and K. Steiglitz. 1982. Combinatorial
   Science 250: 245–250.                                                       optimization: Algorithms and complexity. Prentice-Hall, EngleWood
Dantzig, G.B., R. Rulkerson, and S.M. Johnson. 1954. Solution of a             Cliffs, NJ.
   large-scale traveling salesman problem. Operations Res.                 Reinelt, G. 1991. TSPLIB—A traveling salesman problem library.
   2: 393–410.                                                                 ORSA J. Comput. 3: 376–384.
Deloukas, P., G.D. Schuler, G. Gyapay, E.M. Beasley, C. Soderlund, P.      ———. 1994. The traveling salesman: Computational solutions for TSP
   Rodriguez-Tomé, L. Hui, T.C. Matise, K.B. McKusick, J.S.                   applications. Lecture Notes in Computer Science, vol. 840.
   Beckmann et al. 1998. A physical map of 30,000 human genes.                 Springer Verlag, Berlin, Germany.
   Science 282: 744–746.                                                   Schuler, G.D. 1997. Sequence mapping by electronic PCR. Genome
Dib, C., S. Faure, C. Fizames, D. Samson, N. Drouot, A. Vignal, P.             Res. 7: 541–550.
   Millasseau, S. Marc, J. Hazan, E. Seboun, M. Lathrop, G. Gyapay,        Slonim, D., L. Kruglyak, L. Stein, and E. Lander. 1997. Building
   J. Morissette, and J. Weissenbach. 1996. A comprehensive genetic            human genome maps with radiation hybrids. J. Comp. Biol.
   map of the human genome based on 5,264 microsatellites.                     4: 487–504.
   Nature 380: 152–154.                                                    Stewart, E.A., K.B. McKusick, A. Aggarwal, E. Bajorek, S. Brady, A.
Dunham, I., N. Shimizu, B.A. Roe, S. Chissoe, I. Dunham, A.R. Hunt,            Chu, N. Fang, D. Hadley, M. Harris, S. Hussain et al. 1997. An
   J.E. Collins, R. Bruskiewich, D.M. Beare, M. Clamp et al. 1999.             STS-based radiation hybrid map of the human genome. Genome
   The DNA sequence of human chromosome 22. Nature                             Res. 7: 422–433.
   402: 489–495.                                                           Stringham, H.M., M. Boehnke, and K. Lange. 1999. Point and
Garey, M.R. and D.S. Johnson. 1979. Computers and intractability: A            interval estimates of marker location in radiation hybrid
   guide to the theory of NP-Completeness. Freeman, San Francisco,             mapping. Am. J. Hum. Genet. 65: 545–553.
   CA.                                                                     Walter, M.A., D.J. Spillett, P. Thomas, J. Weissenbach, and P.N.
Goss, S.J. and H. Harris. 1975. New method for mapping genes in                Goodfellow. 1994. A method for constructing radiation hybrid
   human chromosomes. Nature 255: 680–684.                                     maps of whole genomes. Nat. Genet. 7: 22–28.
Gusfield, D. 1997. Algorithms on strings, trees, and sequences.
   Cambridge University Press, Cambridge, UK.
Gyapay, G., K. Schmitt, C. Fizames, H. Jones, N. Vega-Czarny, D.           Received August 3, 1999; accepted in revised form January 6, 2000.




364     Genome Research
           www.genome.org


--- PAGE BREAK ---


