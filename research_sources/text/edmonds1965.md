# Paths, trees, and flowers

> Citation key: `edmonds1965`  
> DOI: 10.4153/CJM-1965-045-4  
> Download URL: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/08B492B72322C4130AE800C0610E0E21/S0008414X00039419a.pdf/div-class-title-paths-trees-and-flowers-div.pdf  
> SHA-256: `c78f3a30df780ffbcf80c81221e4d84343a98a3c1abdbc9dd00eb343b3e1ff85`  

---

                                    PATHS, TREES, AND FLOWERS
                                                         JACK E D M O N D S


              1. Introduction. A graph G for purposes here is a finite set of elements
           called vertices and a finite set of elements called edges such that each edge
           meets exactly two vertices, called the end-points of the edge. An edge is said
           to join its end-points.
              A matching in G is a subset of its edges such that no two meet the same
           vertex. We describe an efficient algorithm for finding in a given graph a match-
           ing of maximum cardinality. This problem was posed and partly solved by
           C. Berge; see Sections 3.7 and 3.8.
              Maximum matching is an aspect of a topic, treated in books on graph
           theory, which has developed during the last 75 years through the work of
           about a dozen authors. In particular, W. T. Tutte (8) characterized graphs
           which do not contain a perfect matching, or 1-factor as he calls it—that is a
           set of edges with exactly one member meeting each vertex. His theorem
           prompted attempts at finding an efficient construction for perfect matchings.
              This and our two subsequent papers will be closely related to other work on
           the topic. Most of the known theorems follow nicely from our treatment,
           though for the most part they are not treated explicitly. Our treatment is
           independent and so no background reading is necessary.
             Section 2 is a philosophical digression on the meaning of "efficient algorithm."
           Section 3 discusses ideas of Berge, Norman, and Rabin with a new proof of
           Berge's theorem. Section 4 presents the bulk of the matching algorithm.
           Section 7 discusses some refinements of it.
              There is an extensive combinatorial-linear theory related on the one hand
           to matchings in bipartite graphs and on the other hand to linear programming.
           It is surveyed, from different viewpoints, by Ford and Fulkerson in (5) and
           by A. J. Hoffman in (6). They mention the problem of extending this relation-
           ship to non-bipartite graphs. Section 5 does this, or at least begins to do it.
           There, the Kônig theorem is generalized to a matching-duality theorem for
           arbitrary graphs. This theorem immediately suggests a polyhedron which in a
           subsequent paper (4) is shown to be the convex hull of the vectors associated
           with the matchings in a graph.
              Maximum matching in non-bipartite graphs is at present unusual among
           combinatorial extremum problems in that it is very tractable and yet not of
           the "unimodular" type described in (5 and 6).


             Received November 22, 1963. Supported by the O.N.R. Logistics Project at Princeton
           University and the A.R.O.D. Combinatorial Mathematics Project at N.B.S.

                                                                    449




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---

                        450                                              JACK EDMONDS

                           Section 6 presents a certain invariance property of the dual to maximum
                        matching.
                           In paper (4), the algorithm is extended from maximizing the cardinality
                        of a matching to maximizing for matchings the sum of weights attached to the
                        edges. At another time, the algorithm will be extended from a capacity of one
                        edge at each vertex to a capacity of dt edges at vertex vt.
                           This paper is based on investigations begun with G. B. Dantzig while at
                        the RAND Combinatorial Symposium during the summer of 1961. I am
                        indebted to many people, at the Symposium and at the National Bureau of
                        Standards, who have taken an interest in the matching problem. There has
                        been much animated discussion on possible versions of an algorithm.

                           2. Digression. An explanation is due on the use of the words "efficient
                        algorithm." First, what I present is a conceptual description of an algorithm
                        and not a particular formalized algorithm or "code."
                           For practical purposes computational details are vital. However, my
                        purpose is only to show as attractively as I can that there is an efficient
                        algorithm. According to the dictionary, "efficient" means "adequate in opera-
                        tion or performance." This is roughly the meaning I want—in the sense that
                        it is conceivable for maximum matching to have no efficient algorithm. Perhaps
                        a better word is "good."
                           I am claiming, as a mathematical result, the existence of a good algorithm
                        for finding a maximum cardinality matching in a graph.
                           There is an obvious finite algorithm, but that algorithm increases in difficulty
                        exponentially with the size of the graph. It is by no means obvious whether
                        or not there exists an algorithm whose difficulty increases only algebraically
                        with the size of the graph.
                           The mathematical significance of this paper rests largely on the assumption
                        that the two preceding sentences have mathematical meaning. I am not
                        prepared to set up the machinery necessary to give them formal meaning, nor
                        is the present context appropriate for doing this, but I should like to explain
                        the idea a little further informally. It may be that since one is customarily
                        concerned with existence, convergence, finiteness, and so forth, one is not in-
                        clined to take seriously the question of the existence of a better-than-finite
                        algorithm.
                           The relative cost, in time or whatever, of the various applications of a
                        particular algorithm is a fairly clear notion, at least as a natural phenomenon.
                        Presumably, the notion can be formalized. Here "algorithm" is used in the
                        strict sense co mean the idealization of some physical machinery which gives
                        a definite output, consisting of cost plus the desired result, for each member of
                        a specified domain of inputs, the individual problems.
                           The problem-domain of applicability for an algorithm often suggests for
                        itself possible measures of size for the individual problems—for maximum
                        matching, for example, the number of edges or the number of vertices in the




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---

                                               PATHS, TREES, AND FLOWERS                 451

         graph. Once a measure of problem-size is chosen, we can define FA(N) to be
         the least upper bound on the cost of applying algorithm A to problems of size N.
            When the measure of problem-size is reasonable and when the sizes assume
         values arbitrarily large, an asymptotic estimate of FA(N) (let us call it the
         order of difficulty of algorithm A) is theoretically important. It cannot be rigged
         by making the algorithm artificially difficult for smaller sizes. It is one criterion
         showing how good the algorithm is—not merely in comparison with other
         given algorithms for the same class of problems, but also on the whole how
         good in comparison with itself. There are, of course, other equally valuable
         criteria. And in practice this one is rough, one reason being that the size of a
         problem which would every be considered is bounded.
            It is plausible to assume that any algorithm is equivalent, both in the
         problems to which it applies and in the costs of its applications, to a * 'normal
         algorithm" which decomposes into elemental steps of certain prescribed types,
         so that the costs of the steps of all normal algorithms are comparable. That is,
         we may use something like Church's thesis in logic. Then, it is possible to ask:
         Does there or does there not exist an algorithm of given order of difficulty for
         a given class of problems?
            One can find many classes of problems, besides maximum matching and its
         generalizations, which have algorithms of exponential order but seemingly
         none better. An example known to organic chemists is that of deciding whether
         two given graphs are isomorphic. For practical purposes the difference between
         algebraic and exponential order is often more crucial than the difference
         between finite and non-finite.
            It would be unfortunate for any rigid criterion to inhibit the practical
         development of algorithms which are either not known or known not to con-
         form nicely to the criterion. Many of the best algorithmic ideas known today
         would suffer by such theoretical pedantry. In fact, an outstanding open
         question is, essentially: "how good" is a particular algorithm for linear pro-
         gramming, the simplex method? And, on the other hand, many important
         algorithmic ideas in electrical switching theory are obviously not "good" in
         our sense.
            However, if only to motivate the search for good, practical algorithms, it
         is important to realize that it is mathematically sensible even to question their
         existence. For one thing the task can then be described in terms of concrete
         conjectures.
            Fortunately, in the case of maximum matching the results are positive.
         But possibly this favourable position is very seldom the case. Perhaps the
         twoness of edges makes the algebraic order for matching rather special in
         comparison with the order of difficulty for more general combinatorial extre-
         mum problems (cf. 3).
            An upper bound on the order of difficulty of the matching algorithm is n4,
         where n is the number of vertices in the graph. The algorithm consists of
         "growing" a number of trees in the graph—at most n—until they augment or




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---

                         452                                             JACK EDMONDS

                        become Hungarian. A tree is grown by branching from a vertex in the tree to
                        an edge-vertex pair not yet in the tree—at most n times. Such a branching
                        may give rise to a back-tracing through at most n edge-vertex pairs in the
                        tree in order to relabel some of them as forming a blossom or an augmenting
                        path. At each of these three levels there may be other labelling work involved—
                        but it is majorized by the work already cited. The work of identifying and
                        labelling the vertex at the other end of some edge to a given vertex need not
                        increase more than linearly with n.
                           An upper bound on the order of magnitude of memory needed for the
                        algorithm is n2—the same order of magnitude of memory used to store the
                        graph itself.

                            3. Alternating paths.
                            3.0. A subgraph of graph G is a graph consisting of a subset of vertices in G
                         and a subset of edges in G under the same incidences which hold for them in G.
                         A non-empty graph G is called connected if there is no pair of non-empty
                         subgraphs of G such that each vertex of G and each edge of G is contained in
                         exactly one of the subgraphs. The vertices and edges of any graph partition
                         uniquely into zero or more connected subgraphs, called its components.
                            Maximum, minimum, and odd will refer to cardinality unless otherwise
                         stated.
                          3.1. The graph E, formed from a set E of edges in G, is the subgraph of G
                        consisting of edges E and their end-points. Any graph H, unless it has a single-
                        vertex component, is formed by its edges. Thus in some contexts it causes no
                        confusion to make no explicit distinction between a graph and its edge-set.
                        In particular, a matching in G may be thought of as a subgraph of G whose
                        components are distinct edges. The sum of two sets D and E is commonly
                        defined as D + E = (D — E) \J (E — D). The sum D + E of two graphs
                        D and E, formed by edge-sets D and E, is defined to be the graph formed by
                        the edge-set D + E.
                            3.2. There are two other kinds of subtraction for graphs besides the set-
                         theoretic difference used above. With these we must distinguish between a
                         subgraph and the edges which form it. Where G is a graph and £ is a set of
                         edges, G — E is the subgraph of G consisting of all the vertices of G and the
                         edges of G not in E. For two graphs G and H, G — H is the subgraph of G
                         consisting of the vertices of G not in H and the edges of G not meeting vertices
                         of H.
                            Graph G VJ H (graph G C\ H) consists of the union (intersection) of the
                         vertex-sets and the edge-sets of graphs G and H, with incidences in G VJ H
                          (graph G O H) the same as in G and H. We may also take the intersection or
                         union of a graph with a set of edges to get, respectively, a set of edges or a
                         graph. In the latter case the end-points of the edges being adjoined to the




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---

                                                  PATHS, TREES, AND FLOWERS             453

            graph must be specified. We shall have occasion to give the same edge different
            end-points in different graphs.
              3.3. A circuit B in graph G is a connected subgraph in which each vertex of
           B meets exactly two edges of B. A (simple) path P in G is either a single vertex
           (joining itself to itself) or else a connected subgraph whose two end-points
           each meet one edge, an end-edge, of P and whose other vertices each meet
           two edges of P. A path is said to join its end-points.
             3.4. For the pair (G, M), where M is a matching in G, a vertex is called
           exposed if it meets no edge of M. Let M denote the edges of G not in M. Define
           an alternating path or alternating circuit, P , in (G, M) to be such that one edge
           in M C\ P and one edge in M C\ P meets each vertex of P , except the end-
           points in the case of a path. Several authors, beginning with J. Peterson in
           1891, have used alternating paths to prove the existence of * 'factors" in certain
           kinds of graphs.
              3.5. For any two matchings M\ and M2 in G, the components of the subgraph
           formed by Mi + M2 are paths and circuits which are alternating for (G, Mi)
            and for (G, M2). Each path end-point is exposed for either Mi or M2.
             A vertex of G meets no more than one edge, each, of Mi and M2—and thus
           no more than two edges of Mi + M2, one in Mi Pi M2 and one in M2 P Mlm
           An end-point v of a path in graph Mi + M2, meeting an end-edge in Mi P M2i
           say, meets no other edge of Mi. Hence, if an edge of M2 meets v, it does not
           belong to Mi and so it does belong to Mi + M2. But then v is not an end-point.
           Therefore v is exposed for M2. This completes the proof.
             3.6. An alternating path A in (G, M) joining two exposed vertices contains
           one more edge of M than of M. M + A is a matching of G larger than M by
           one. Such a path is called augmenting. Thus matching M is not maximum if
           (G, M) contains an augmenting path. The converse also holds:
              3.7 (Berge, 1). A matching M in G is not of maximum cardinality if and only
           if (G, M) contains an alternating path joining two exposed vertices of M.
            If M2 is a larger matching than M, some component of graph M + M2
           must contain more M2-edges than M. By 3.5, such a component is an aug-
           menting path for (G, M).
              3.8. Berge proposed searching for augmenting paths as an algorithm for
           maximum matching. In fact, he proposed to trace out an alternating path
           from an exposed vertex until it must stop and, then, if it is not augmenting, to
           back up a little and try again, thereby exhausting possibilities.
              His idea is an important improvement over the completely naive algorithm.
           However, depending on what further directions are given, the task can still
           be one of exponential order, requiring an equally large memory to know when
           it is done.




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---

                      454                                             JACK EDMONDS


                        Norman and Rabin (7) present a similar method for finding in G a minimum
                      cover-by-edges, C, a minimum cardinality set of edges in G which meets every
                      vertex in G. The Berge-Norman-Rabin theorem (2) is generalized in (3), but a
                      corresponding generalization of the algorithm presented here in Section 4 is
                      unknown.
                         3.9. Norman and Rabin also show that the maximum matching problem and
                      the minimum cover-by-edges problem are equivalent.
                         Assuming every vertex meets an edge, the minimum cardinality of a cover
                      of the vertices in G by a set of edges equals the minimum cardinality of a cover
                      of the vertices in G by a set of edges and vertices, where a vertex is regarded
                      as covering itself. By replacing edges by vertices or vice versa, one can go
                      back or forth between a minimum cover by a set of edges and a minimum
                      cover by a set of edges and vertices, where the latter set consists of a maximum
                      matching together with its exposed vertices.

                         4. Trees and flowers.
                         4.0. A tree may be defined as (1) a graph T every pair of whose vertices is
                      joined by exactly one path in T\ (2) inductively, as either a single vertex or
                      else the union of two disjoint trees together with an edge which has one end-
                      point in each; (3) as a connected graph with one more vertex than edges; and
                      so on.

                         4.1. An alternating tree J is a tree each of whose edges joins an inner vertex
                      to an outer vertex so that each inner vertex of J meets exactly two edges of / .
                      An alternating tree contains one more outer vertex than inner vertices. This follows
                      from the third definition of tree by regarding each inner vertex with its two
                      edges as a single edge joining two outer vertices.

                        4.2. For each outer vertex v of an alternating tree J there is a unique maximum
                      matching of J which leaves v exposed and the only exposed vertex in J. Every
                      maximum matching of J is one of these.

                         Definition (2) of tree can be strengthened to the statement that a tree minus
                      any one of its edges is two trees. Thus / minus any one of its inner vertices,
                      say u, is two alternating trees. One of these, J i , contains v as an outer vertex.
                      Assume inductively that J\ can be matched uniquely so only v is exposed and
                      that J 2 , the other subtree, can be matched uniquely so only the vertex v2,
                      joined in J to u by edge e2j is exposed. Then the union of e2 and these two
                      matchings is a matching of / which leaves only v exposed. Since every edge of
                      / has one inner and one outer end-point, every maximum matching leaves only
                      an outer vertex exposed.

                        4.3. A planted tree, J = J(M), of G for matching M is an alternating tree in
                      G such that M P\ / is a maximum matching of / and such that the vertex r




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---

                                                  PATHS, TREES, AND FLOWERS                455

            in / which is exposed for M C\ J is also exposed for M. That is, all matching
            edges which meet / are in J. Vertex r is called the root of J(M).

              In planted tree J(M) every alternating path P(M), which has outer vertex v
            and the matching edge to v at one of its ends, is a subpath of the alternating path
            PV(M) in J(M) which joins v to the root r.
               For k > 1, assume that P^k-i is the unique path P(M) which contains
            2& — 1 edges and assume that at its non-y end it has an inner vertex uk and a
            matching edge. Then P2k, consisting of P2/c-i together with the unique non-
            matching edge in J which meets uk, is the unique path P{M) with 2k edges.
            It has outer vertices vk and v at its two ends. If vk ^ r, then P2yfc+i> consisting
            of Pik together with the unique matching edge which meets vk, is the unique
            path P(M) with 2& + 1 edges. It has an inner vertex uk+1 and a matching
            edge at its non-y end. Since our assumption is true for k = 1 and since k cannot
            become infinite, the theorem follows by induction.
              We define a stem in (G, M) as either an exposed vertex or an alternating path
            with an exposed vertex at one end and a matching edge at the other end. The
            exposed vertex and the vertex at the other end are, respectively, the root and
            the tip of the stem. The preceding theorem tells us that (1) no trial-and-error
            search is required to find the path in J from any of its vertices back to the root
            and (2) the path Pv in / joining any outer vertex v to the root of J is a stem.

              4.4. An augmenting tree, JA = JA(M), in (G, M) is a planted tree J(M) plus
            an edge e of G such that one end-point of e is an outer vertex v\ of J and the
            other end-point V2 is exposed and not in J. The path in JA which joins V2 to the
            root of J is an augmenting path. This follows immediately from (4.3).

               4.5. For each vertex b of an odd circuit B there is a unique maximum
            matching of B which leaves b exposed. A blossom, B = B(M), in (G, M) is an
            odd circuit in G for which M C\ B is a maximum matching in B with say
            vertex b exposed for M C\B. A flower, F = F (M), consists of a blossom and
            a stem which intersect only at the tip of the stem (the vertex b).
               A flowered tree, JF, in (G, M) is a planted tree / plus an edge e of G which
            joins a pair of outer vertices of J. The union of e and the two paths which join
            its outer-vertex end-points to the root of J is a flower, F.
               Let Vi and V2 be these outer vertices, and P i and P2 be the paths in / joining
            them to r. We have seen that Pi and P 2 are stems (which are easily recovered
            from J). Since they intersect in at least r and since the path in J joining r to
            any other vertex is unique, Pb = P i Pi P 2 is an alternating path with an end
            at r. If its other end-point, say b, were inner, it would be distinct from r, vi,
            and V2- Thus r would be distinct from v\ and t;2, and b would meet three different
            edges of / , one in P 6 , one in P i not in Ph, and one in P 2 not in P 6 . But an
            inner vertex meets only two edges in the tree. Therefore b is outer and Pb is a
            stem. Thus P / = P i — (P 6 — b) and P2 = P 2 — (P& — b), unless one is a




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---

                    456                                              JACK EDMONDS

                    vertex vi = b or v2 = b, h a v e n o n - m a t c h i n g edges a t their ô-ends a n d m a t c h i n g
                    edges a t their outer ends. I t follows in a n y case t h a t B = P / U P2 U g is a
                    circuit with only b exposed for M C\B, a n d t h u s B is a blossom with £ ô as
                    its stem.

                       4.6. A Hungarian tree H in a. graph G is an a l t e r n a t i n g tree whose o u t e r
                     vertices are joined b y edges of G only to its inner vertices.

                        4.7. For a matching M in a graph G, an exposed vertex is a planted tree. Any
                     planted tree J(M) in G can be extended either to an augmenting tree, or to a
                    flowered tree, or to a Hungarian tree (merely b y looking a t most once a t each of
                     the edges in G which join vertices of the final t r e e ) .

                        An exposed vertex satisfies the definition of planted tree. Suppose we are
                     given a planted tree J a n d a set D (perhaps e m p t y ) of edges in G which are
                     n o t in J b u t which join outer to inner vertices of / . (1) If no o u t e r vertex of J
                     meets an edge n o t in D \J J, then / is H u n g a r i a n . Suppose outer vertex v\
                     meets an edge e n o t in D VJ / , whose other end-point is, say, v2. (2) If v2 is an
                     inner vertex of J , we can enlarge D b y adjoining e. (3) If v2 is an o u t e r vertex
                     of J , then e U / is a flowered tree. (4) If v2 is exposed a n d n o t in / , then
                     e VJ J is an a u g m e n t i n g tree. (5) Finally, if v2 is n o t exposed a n d n o t in J ,
                     then the iVf-edge e2 which meets v2 is n o t in / , a n d t h u s z/3, t h e other end-point
                     of e2} is not in / b y the definition of p l a n t e d tree. Therefore, in this case we can
                     extend / to a larger planted tree with new inner vertex v2 a n d new o u t e r v e r t e x
                     v-s b y adjoining edges e and e2. For a n y J a n d D, one of t h e five cases holds.
                     Therefore b y looking a t a n y edge in G a t most once, we can reach one of the
                     three cases described in the theorem, because t h e other two cases, (2) a n d (5),
                     consume edges and G is finite.

                         4.8. T h e algorithm which is being constructed is efficient because it does n o t
                     require tracing m a n y various combinations of t h e same edges in order to find
                     an a u g m e n t i n g p a t h or to d e t e r m i n e t h a t there are none. In fact we accomplish
                     one or the other w i t h o u t ever looking again a t the edges encountered in process
                     (4.7), except to pick o u t from the tree t h e blossom or the a u g m e n t i n g p a t h
                     when case (3) or (4) occurs. W e see from (4.3) a n d (4.5) how easy it is to
                     retrieve t h e blossom or the p a t h . W h e n flowers arise we " s h r i n k " t h e blossoms,
                     a n d so if an a u g m e n t i n g p a t h arises later, it will be in a " r e d u c e d " graph.
                     However, only one other v e r y simple kind of t a s k t r a n s l a t e s t h e a u g m e n t a t i o n
                     to (G, M) itself. T h a t t a s k is to expand a shrunken blossom t o an odd circuit
                     a n d find the m a x i m u m m a t c h i n g of t h e odd circuit which leaves a certain
                     vertex exposed. Actually, we shall find in (7.3) t h a t it is desirable to leave odd
                     circuits shrunk while looking in the reduced graph for as m a n y successive aug-
                     m e n t a t i o n s as possible since t h e y are all reflected in a u g m e n t a t i o n s of (G, M).

                       4.9. For H, a subgraph of G, G is t h e disjoint union (G - H) U ÔH U H+,
                     where 5H is the set of the edges with one end-point in H a n d one end-point in




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---

                                                 PATHS, TREES, AND FLOWERS               457
           G — H, and where H+ = G — (G — H) is the subgraph consisting of H and
           all edges of G with both end-points in H. When H is connected, shrinking H
           means constructing the new graph G/H = (G — H) \J 8H U h by regarding
           H+ as a single new vertex, h = H/H, which meets the edges 8H = 8h. The
           end-points in G — H of the edges 8H do not change.
              4.10. If B is an odd circuit in G, then b' = B/B is called a pseudovertex of
           G IB. To expand b' means to recover G from G/B. The algorithm, after it
           expands a pseudovertex 6', will make use of the circuit B. In general, finding
           a "Hamiltonian" circuit in a graph B+ is difficult. Therefore, when the
           algorithm shrinks B to form G/B, it should remember circuit B as having
           effected the shrinking. Thus we call circuit B in G (rather than B+) the expansion
           of b''. In formal calculation shrinking B in G is an easy operation. Essentially,
           just assign all the vertices and edges of B SL label, b', and then, until b' is ex-
           panded by erasing these labels, ignore any distinction between vertices labelled
           b' and ignore edges joining them to each other.
              Where M is a matching set of edges in G, M/B is defined as M C\ (G/B).
           Clearly, if B is a blossom for (G, M), then M/B is a matching of G/B.
              4.11. Let Go = G, Gi = Gi-\/Bu and 6j = Bi/B i for z = 1, . . . , n, where
           5 j is an odd circuit in graph Gt-i. We inductively define the pseudovertices
            (with respect to G) of Gk (k = 1, . . . , n) to be ^ together with the pseudo-
           vertices in Gt-i — Bk = Gk — £#. Of course not every bu i < &, will be a
           pseudovertex of G^ because some will have been absorbed into others. The
           order in which the pseudovertices of a Gk arise is immaterial. That is, the order
           in which the odd circuits Bt are shrunk is immaterial except in so far as one
           shrunken B t is a vertex in another B t. Thus we can expand any pseudovertex
           bj of Gk to obtain a graph Gkj for which Gkj/Bj = Gk. The pseudovertices of
           Gkj (with respect to G) are the pseudovertices in Bj together with the pseudo-
           vertices in Gk — bj] that is, graph Gkj can be obtained from G by shrinking
           in a proper order the odd circuits which wTere absorbed into these pseudo-
           vertices. On the other hand, we do not expand a vertex bh in Bk until vertex
           b,c is expanded.
              There is a partial order on the b/s defined by the transitive completion of
           the relation bh < bk where bh is a vertex of Bk. (It is a special kind of partial
           ordering because each bh is a vertex of at most one Bk.) There is a partial order
           on the sets, Sa, Sp, . . . , of mutually incomparable b/s, where Sa < Sp when
           every member of Sa is less than or equal to some member of S$. Evidently
           there is a unique family of graphs, Ga, Gp, . . . , which include the G/s and G.
           They correspond 1-1 to the sets, Sa, Spy . . . , so that the pseudovertices of Ga
           are Sa, etc. We have Sa < Sp if and only if Gp can be obtained from G« by
           shrinking certain Bu those for which bt is less than or equal to some member
           of Sp and not less than or equal to any member of Sa. Graph G corresponds
           to the empty set and Gn corresponds to the set of b/s which are maximal with
           respect to their partial order.




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---

                      458                                             JACK EDMONDS

                         The complete expansion of a pseudovertex bt is the subgraph
                                                              U+ =                  G-(G-U)CG
                     where U consists of all vertices of G absorbed into bt by shrinking.
                       4.12. Where B is the blossom of a flower F for (G, M), M is a maximum
                     matching of G if and only if M/B is a maximum matching of G/B.
                       4.13. Where blossom B is in JF, a planted flowered tree for (G, M), JF/B is a
                     planted tree for (G/B, M/B). It contains B/B as an outer vertex. Its other outer
                     and inner vertices are respectively those of JF which are not in B.
                       Theorem (4.13) follows easily from (4.5). We separate the two converse
                     statements of Theorem (4.12) into slightly stronger statements, (4.14) and
                     (4.15).
                        4.14. Where B is any odd circuit in G, for every matching Mi of G/B there
                     exists a maximum matching MB of B such that M — Mi \J MB is a matching
                     for G.
                        Since any matching Mi of G/B contains at most one edge meeting B/B. the
                     edges Mi in G meet at most one vertex, say bi, of B. Therefore the desired MB
                     is the maximum matching of B which leaves bi exposed. Since the cardinality
                     \MB\ of MB is constant, any augmentation of M\ yields a corresponding
                     augmentation of M. Therefore, the "only if" part of (4.12) is proved.
                        Applying the above matching operation to successive expansions of pseudo-
                     vertices into odd circuits we have:
                        Where P is the complete expansion of a pseudovertex p in G2, where GL is the
                     graph obtained from G2 by completely expanding p, and where M2 is any matching
                     of G:2, there exists a matching MP of P leaving exactly one exposed vertex m P
                     such that MP \J M2 is a matching of G\. Thus since \MP\ is constant, any
                     augmentation in G2 yields a corresponding augmentation in Gi.
                        4.15. For (G, M), let P be a subgraph such that (1) M Pi P leaves exactly one
                     exposed vertex in P , (2) M/P is a maximum matching of G/P, and (3) p = P/P
                     is the tip of a stem Svfor (G/P, M/P). Then M is a maximum matching of G.
                         The edges of SP form in G a stem, S, for (G, M). (In case Sp has no edges,
                      take 5 to be the vertex in P exposed for M.) Compare M' = AI -\- S and
                      M'/P with M and M/P. The definition of stem implies that M' is a matching
                      of G with \M'\ = \M\ and that the exposure of the root of S is changed to the
                      exposure of the tip of S. Similarly M'/P = M/P + Sp is a matching of G/P
                      with \M'/P\ = | M / P | and with vertex p exposed. Because the cardinalities
                      do not change, it is sufficient to show that M' is maximum in G if Mf/P is
                      maximum in G/P.
                         Using (3.7), if M' is not maximum, G contains an augmenting path
                      A = A(M'). If A contains no vertices of P , then it is also an augmenting




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---

                                                 PATHS, TREES, AND FLOWERS                   459

           path for M1 IP in G/P. Otherwise, because P contains only one exposed vertex
           for M', at least one of the ends of A is at an exposed vertex U\ not in P . There
           is a unique subpath A\ of A with one end-point at U\ and containing only one
           vertex pi of P , at its other end. The only difference between A\ and A\/P =
           (A i \J P)/P is that pi is replaced by p, which is exposed for Mf/P. Thus
           A\/P is an augmenting path for Mf'/P and so M'/P is not maximum. The
           theorem is proved.
               The theorem extends as follows:
             For (G, M), let Pi, . . . , Pn be a family of disjoint subgraphs in G such that
           (1) M r\ Pi leaves exactly one exposed vertex in Pu (2) Mn = M Pi Gn is a
           maximum matching of Gn = G / P i / . . . /Pn, and (3) vertices Pi/Pt of Gn are
           outer vertices in a planted tree Jnfor (Gn, Mn). Then M is a maximum matching
           ofG.
              We may assume that the indices order the P * / P / s so that (for
           k = 1, . . . , n — 1) those from 1 through k are contained in a planted subtree
           JJC of Jn not containing those from k + 1 through n. Hence the theorem follows
           by induction after proving that Mn-\ = M C\ Gn-\ is a maximum matching of
           Gn-\ = G / P i / . . . /Pn-i- Since every outer vertex of Jn is the tip of a stem
           in Gn, this follows from the last theorem.
              4.16. Theorems (4.7) and (4.13) show how by branching a planted tree out
           from an exposed vertex of (C7, M) and shrinking blossoms Bt when they are
           encountered, we eventually obtain in a graph Gk = G/B\/ . . . /Bk either a
           tree with an augmenting path or a Hungarian tree. An augmenting path
           admits an augmentation of matching Mk = M f~\Gk according to (3.7), and
            (4.14) shows how this induces an augmentation of matching ikf^-i = M f~\ Gk-i
           and so on back through M. On the other hand, when a Hungarian tree J is
           obtained, submatching ( / U BkVJ . . . \J Bi) C\ M of (G, M) cannot be im-
           proved and so this part of G is freed from further consideration. This follows
           immediately from (4.15) and the next theorem, (4.17), where Gk is denoted
           simply as G.
             4.17. Let J be a Hungarian tree in a graph G. A matching Mi of G — J is
           maximum in G — J if and only if Mi together with any maximum matching
           Mj of J is a maximum matching of G.
             Since J and G — J are disjoint, if there exists a matching M / ol G — J
           which is larger than Mu then Mi \J Mj is a larger matching of G than
           Mi U M j . Conversely, suppose Mi is maximum for G — J. Let
                                                    M' = Mi' U MjKJ                     M/
           be an arbitrary matching of G where Mi d G — J, where M/ C / , and
           where MT Pi ((<? - J) U J) is empty. Then | M / | < |Afi|. Every edge in M7
           meets at least one inner vertex of / ; that is, where V C I {J) is the set of the
           inner vertices met by MT, \MT\ < \I'\. The graph J — V consists of \V\ + 1




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---

                       460                                             JACK EDMONDS


                      disjoint alternating trees whose inner vertices together are I (J) — / ' . There-
                      fore, since the maximum matching cardinality of an alternating tree equals
                      the number of its inner vertices, \M/\ < \I(J) — If\. Adding the three in-
                      equalities gives \M'\ < \Mi\ + \I(J)\ = |Mi U Mj\. So the theorem is proved.
                         4.18. The matching M of G = G°, to begin with, may be empty. If it leaves
                      any exposed vertices, then the process (4.16) operates with respect to one of
                      them. Either it produces an augmentation of M by one edge, thus disposing of
                      two exposed vertices, or it reduces the possible domain for augmenting M to
                      a subgraph G1 = Gk — J of G, containing one less exposed vertex and con-
                      taining only edges and vertices not previously considered. Successive applica-
                      tion of (4.16) may reduce the consideration of M to a subgraph Gl of G and
                      reveal there an augmentation of M. After augmenting in G\ obtaining a
                      larger M for G with two less exposed vertices in G\ (4.16) operates again in
                      G\ never returning to the matching in the rest of G.
                         4.19. Repeated application of (4.18) reduces the domain in question to a
                      Gn containing no exposed vertices. Then we know that we have a maximum
                      matching; let us still call it M, with n exposed vertices in G. Thus the construc-
                      tion of an algorithm for finding a maximum cardinality matching in a graph
                      is complete. Often the last application of (4.18) is unnecessary. For verifying
                      maximality, the algorithm may as well stop when it reduces the domain to a
                      Gn~~l containing one exposed vertex, since two exposed vertices are necessary
                      in order to augment. However, for theoretical purposes it is convenient to have
                      the algorithm grow a tree from each exposed vertex of the final, maximum
                      matching.
                         4.20. We may define an alternating forest to be a family of disjoint alternating
                      trees and a planted forest in (G, M) to be a family of disjoint planted trees in
                       (G, M). A dense planted forest is one which contains all the exposed vertices
                      of (G, M). The family of exposed vertices, itself, is a dense planted forest. The
                      algorithm works as well by growing a dense planted forest all at once, rather
                      than one tree at a time.
                         It is appropriate then to define augmenting forest (flowered forest) to be a
                      planted forest plus an edge e of G whose end-points are outer vertices of
                      different trees (of the same tree) of the planted forest.
                         A Hungarian forest in G is defined similarly to Hungarian tree, replacing
                      the word "tree" by ''forest." Notice that the trees of a Hungarian forest are
                      not necessarily Hungarian trees—an outer vertex of one tree may be joined
                      by an edge of G to an inner vertex of another tree in the forest.
                         The theorems on trees presented in this section are essentially the same for
                      forests.

                          5. The dual to matching.
                          5.0. A bipartite graph K is one in which every circuit contains an even




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---

                                                  PATHS, TREES, AND FLOWERS                    461

           number of edges. This condition, that K contains no odd circuits, is equivalent
           to being able to partition the vertices of K into two parts so that each edge of
           K meets exactly one vertex in each part. The well-known Kônig theorem
           states:

               For a bipartite graph K, the maximum cardinality of a matching in K equals
            the minimum number of vertices which together meet all the edges of K.
               5.1. The linear programming duality theorem states: If
               (1) x > 0, Ax < c and
               (2) y ^0,ATy>        b,
            for given real vectors b and c and real matrix A, then for real vectors x and y,
                                                     maxx(b,x)        = mmy(c, y)
           when such extrema exist.

             The problems of finding a maximizing vector x and a minimizing vector y
           are called linear programmes, dual to each other.

              5.2. The Kônig theorem is now widely recognized as the instance of (5.1)
           where b and c consist of all ones and A = AK is the zero-one incidence matrix
           of edges (columns) versus vertices (rows) in a bipartite graph K. In view of
           Theorem (5.1) the Kônig theorem is equivalent to the remarkable fact that,
           with b, c, and A as just described, the two linear programmes of (5.1) have
           solutions x and y whose components are zeros and ones whether or not this
           condition is imposed. An elegant theory centres on this phenomenon.
              Graph-theoretic algorithms are well known for so-called assignment, trans-
           portation, and network flow problems (5). These are linear programmes
           which have constraint matrices A that are essentially AK.
              5.3. For a linear programme with an arbitrary matrix A of integers, or even
           of zeros and ones, we cannot say that the extreme values will be assumed, as
           when A = AK, by vectors with integer components. Therefore, in general
           when we impose the condition of integrality on x, the equality of the two
           extrema no longer holds.
              In particular, when the maximum matching problem is extended from bipar-
           tite to general graphs G, a genuine integrality difficulty is introduced. Our
           matching algorithm met it by the device of shrinking blossoms.

              5.4. The matching algorithm yields a generalization of the Konig theorem
           to maximum matchings in G. The new matching duality theorem, in the form
           "maximum cardinality of a matching in G equals minimum of something
           else," is also an instance of linear programming duality.
              It is reasonable to hope for a theorem of this kind because any problem which
           involves maximizing a linear form by one of a discrete set of non-negative
           vectors has associated with it a dual problem in the following sense. The discrete




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---

                      462                                              JACK EDMONDS

                      set of vectors has a convex hull which is the intersection of a discrete set of
                      half-spaces. The value of the linear form is as large for some vector of the
                      discrete set as it is for any other vector in the convex hull. Therefore, the
                      discrete problem is equivalent to an ordinary linear programme whose con-
                      straints, together with non-negativity, are given by the half-spaces. The dual
                      (more precisely, a dual) of the discrete problem is the dual of this ordinary
                      linear programme.
                         For a class of discrete problems, formulated in a natural way, one may hope
                      then that equivalent linear constraints are pleasant even though they are not
                      explicit in the discrete formulation.
                         5.5. Arising from the definition of a matching—no more than one matching
                      edge to each vertex—are the obvious linear constraints that for each vertex
                      v 6 G the sum of the x's corresponding to edges which meet v is less than one.
                      To obtain a maximum cardinality matching, we want to maximize the sum of
                      all the x's, corresponding to edges of G, subject to the additional condition
                      that each x is zero or one.
                         It turns out that maximum matching can be turned into linear programming
                      by substituting for the zero-one condition the additional constraints that the
                      x's are non-negative and that for any set R of 2k + 1 vertices in G(k = 1,2,...)
                      the sum of the x's which correspond to edges with both end-points in R is no
                      greater than k. The former condition on the x's obviously implies the latter
                      since for no matching in G do more than k matching edges have both ends in R.
                         The converse—that subject only to the linear constraints, ^ x{ can be
                      maximized by zeros and ones—is not so obvious, but in view of (5.1) it follows
                      from (5.6), the generalized Kônig theorem.
                         Actually the stronger converse holds—that subject only to these same
                      linear constraints, YLcixu for any real numbers cu can be maximized by
                      zeros and ones. In other words, the polyhedron described by the constraints
                      is, indeed, the convex hull of the zero-one vectors which correspond to match-
                      ings in G. We shall not prove this until we take up maximum weight-sum
                      matching in paper (4). Although the convex-hull notion suggested trying to
                      generalize the Kônig theorem, and although the generalization found does
                      suggest the true convex hull, the success of the first suggestion does not neces-
                      sarily validate the second.

                         5.6. A set consisting of one vertex in G is said to cover an edge e in G if e meets
                      the vertex. The capacity of this set is one. A set consisting of 2k + 1 vertices
                      in G(k = 1, 2, . . .) is said to cover an edge e in G if both end-points of e are
                      in the set. The capacity of this set is k. An odd-set cover of a graph G is a family
                      of odd sets of vertices such that each edge in G is covered by a member of the
                      family.

                         MATCHING-DUALITY THEOREM. The maximum cardinality of a matching in G
                       equals the minimum capacity-sum of an odd-set cover in G.




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---

                                                  PATHS, TREES, AND FLOWERS              463
               It is obvious that the capacity-sum of any odd-set cover in G is at least as
            large as the cardinality of any matching in G, so we have only to prove the
            existence in G of an odd-set cover and a matching for which the numbers are
            equal.
               5.7. The theorem holds for a graph which has a perfect matching M—that
            is, with no exposed vertices—since the odd-set cover consisting of two sets,
            one set containing one of the vertices and the other set containing all the other
            vertices, has capacity-sum equal to \M\. It also holds for a graph which has a
            matching with one exposed vertex. Here the odd-set cover may be taken as
            consisting of one member, the set containing all vertices of the graph. For the
            case of one exposed vertex, an odd-set cover may also be constructed as in
            (5.8) by applying the algorithm to construct a Hungarian tree even though
            it obviously will not result in augmentation.
               5.8. Applying the algorithm to (G, M), where \M\ is maximum, using some
            exposed vertex as root, we obtain a graph Gr containing a maximally matched
            Hungarian tree / , a number of whose outer vertices are pseudo. Let Sj consist
            of all odd sets of the following two types: sets each consisting of one inner
            vertex in / , and sets each consisting of the vertices in the complete expansion
            of one pseudovertex of J.
               The number of edges of M which a member of Sj covers is equal to the
            capacity of the member. Every edge of M not m Gf — J is covered by exactly
            one member of Sj. An edge of G is covered by a member of Sj if and only if
            it is not in G' — J.
               Matching M C\ (Gf — J) is a maximum matching oi G' — J with one less
            exposed vertex than (G, M). Assuming that | M C\ (Gf — J) | equals the
            capacity of an odd-set cover, say S'j, of G' — / , we have that \M\ equals the
            capacity of Sj \J S'j, an odd set cover of G. Theorem (5.6) follows by induction
            on the number of exposed vertices.
              5.9. It is evident from the proof that we may require the minimum odd-set
            cover to have certain other structure—in particular, that each member with
            more than one vertex contain the vertices of at least one odd circuit in G.
            With the latter restriction the theorem becomes a strict generalization of the
            Konig theorem.

               6. Invariance of the dual.
              6.0. For any particular application of the algorithm (4) to G, yielding, say,
            the maximum matching M, we may skip the augmentation steps in (4.16)
            by regarding the augmented matching as being the one already at hand. This
            gives a particular application of (4) to G starting with maximum matching M.
            In the application of the algorithm to (G, M), we can regard all the branchings
            and blossom shrinkings as taking place without subtracting the trees Jt as
            they arise. Thus we obtain from (G, M) a graph G* with a number of pseudo-




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---

                      464                                             JACK EDMONDS


                      vertices which are outer vertices in a sequence {Ji}(i = 1, . . . , n) of disjoint
                      planted trees in G*, one corresponding to each exposed vertex of (G, M). By
                      expanding all the pseudovertices of G* completely, we recover the graph G.
                         6.1. The tree Jt is Hungarian in G* — J\ . . . — Jt-i, but usually not
                      Hungarian in G* because an outer vertex of J{ might be joined to an inner
                      vertex of any other tree with a lower index. Hence the partition of the outer
                      and inner vertices into trees Jt depends on the order of their construction.
                      Also non-matching edges which can occur in each tree are not unique. In
                      general, joining outer to inner vertices of a Jt are many other M edges which
                      would do as well. The particular blossoms which led to the pseudovertices are
                      also fairly arbitrary. And, finally, the maximum matching is far from unique.
                      However, (6.2) will show that the graph G* is uniquely determined by G alone.
                         6.2. For a (G, M) where M is any maximum matching, let G* and \Jt] be
                      obtained from (G, M) by (6.0).
                         (a) The non-pseudo outer vertices of the J / s and the vertices of the pseudovertex
                      complete expansions, all called the outer vertices 0(G) of G, are precisely the vertices
                      of G which are left exposed by some maximum matching of G.
                         (b) The inner vertices of the JYs, called the inner vertices 1(G) of G, are precisely
                      those vertices of G not in O (G) but joined to vertices in O (G).
                         (c) G* is obtained from G by shrinking the connected components of 0(G)+,
                      the subgraph of G consisting of vertices 0(G) and all edges of G joining them.
                        6.3. We have defined vertex families 0(G) and 1(G) in terms of particular Jt.
                      The theorem yields definitions dependent only on G itself.
                        Clearly 0(G*) and I(G*), defined in terms of the Jt in G*, are respectively
                      the outer and the inner vertices of the J\. Notice that the early definitions of
                      inner and outer, for vertices in an alternating tree, are consistent with the
                      definitions for a general graph.
                         6.4. Proof of (6.2), (b) and (c). Let the vertex v* of G* be joined in G* to
                      some outer vertex w* of J\. Then v* is a vertex in some Jh(h < i)y since Jt is
                      Hungarian in G* — J\ — . . . — Ji-\. But y* cannot be an outer vertex of
                      Jh since u* is not inner and since Jn is Hungarian in G* — Ji — . . . — Jh-\-
                      Therefore v* is inner. It follows that each outer vertex u of G is joined only to
                      inner vertices and to other vertices in the complete expansion of its image u*.
                      By construction, each inner vertex is joined to an outer vertex of G. Hence,
                       (b) is true.
                         Since by construction the complete expansion of each outer vertex of G* is
                      connected, it also follows that the connected components of 0(G)+ correspond
                      precisely to outer vertices of G*. Hence, (c) is true.
                        6.5. An outer vertex u of G, by definition, either is identical with or is
                      contained in the complete expansion of some outer vertex u* of, say, J\. For
                      any maximum matching Mt of alternating tree Ju Mt ^J [M C\ (G* — Jt)] is




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---

                                                  PATHS, TREES, AND FLOWERS             465

            a maximum matching of G*, which by (4.14) induces a maximum matching
            M' of G. Let Mt be the one which leaves u* exposed. If u* is pseudo, then by
            (4.14) M' can be chosen so that u is exposed in the expansion. This proves
            half of (6.2), (a).
               6.6. C. Witzgall suggested the following simplified proof of the converse,
            viz. that only the outer vertices are ever exposed for a maximum matching.
            A non-outer vertex v meets an edge e of M. Deleting v and its adjoining edges,
            \J Ji — v is a Hungarian forest in G* — v.
               If v is inner, then the forest is dense in G* — v. Otherwise it is dense in
            G* — v except for one exposed vertex, the other end of e. In either case it
            follows that M — e is a maximum matching of G — v.
               Assume that M' is a maximum matching of G which leaves v exposed. Then
            M' is also a matching of G — v. Since M' is larger than M — e, we have a
            contradiction. This completes the proof of (6.2).

              6.7. The definition of odd-set cover may be expanded (more than necessary
           for Theorem (5.6)) to include the possibility of members which are even sets
           of vertices in G. A set of 2&-vertices has capacity k and covers the edges which
           have both end-points in the set. Then, clearly, Theorem (5.6) still holds for
           this kind of cover.
              With this definition of cover, it follows from the uniqueness of G* that there
           is a unique preferred minimum cover, S*, for any graph G. The one-vertex
           members of 5* are the inner vertices of G*, the other odd members of S*
           correspond to the pseudovertices of G*, and the one even member of S* consists
           of the non-inner, non-outer vertices of G*.

               7. Refinement of the algorithm.

              7.0. Several possibilities for refining the algorithm suggest themselves.
             We could remember an old tree, uprooted by an augmentation, so that when
           a new rooted tree takes on a vertex in it, we can immediately adjoin a piece
           of it to the new tree. This appears not worth doing. A tree is easy to grow,
           easier than selecting from an old tree the piece which may be grafted.

              7.1. A quite useful refinement is to leave the pseudovertices of the old tree
           shrunk until their expansion is necessary. We see from (4.14) that any further
           augmentation of a matching M' in a graph G with pseudovertices yields a
           further augmentation in G just as easily as the first. On the other hand, a
           maximum matching in G', reached after one or more augmentations, does not
           necessarily yield a maximum matching of G. The sufficiency part of (4.12)
           depends on the blossom being part of a flower, whereas the first augmentation
           in G! uproots the stem.
             7.2. However, we may easily observe the circumstance arising in the
           application of the algorithm to (G', Mr) where the shrinkage might hide a




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---

                      466                                              JACK EDMONDS

                      possible augmentation in G. It is where a pseudovertex, say b'> becomes an
                      inner vertex of the planted tree, say Jf = J'(M').

                         In this case, we obtain a graph G" from Gr by expanding b' to an odd circuit
                      B. The edges of J' form in G" a subgraph which we still call J'. The set Mr is
                      also a matching in G". One edge of J' C\ M' has an end-point, say bu in B.
                      One edge of / ' f\ M' has an end-point, say b2, in B. The maximum matching
                      MB of B which is compatible with Mr in G" leaves b\ exposed. The vertices
                      b\ and bi partition B into two paths, P2 even and P\ odd, which join b\ and bo.
                      The graph J" = P2 U J ' is a planted tree in G" for the matching MB W M'.
                         Unless 61 and J2 coincide, P 2 will contain outer vertices of J". These may be
                      joined to vertices not in J" which admit an extension of J", not possible for
                      J" jB = J' C G', to a planted tree with an augmenting path.

                          7.3. If J' C Gr can be extended in G! to a tree with an augmenting path, it
                       does not matter that some of the inner vertices are pseudo because a further
                       augmentation for G is thus determined. If J' with pseudo inner vertex b' can
                       be extended in (G', M') to a flowered tree whose blossom B' contains b\ then
                       bf loses its distinction as an inner vertex. It might as well stay shrunk and be
                       absorbed into the new pseudovertex B'/B' of G'/Bf. In fact, Theorems (4.15)
                       and (4.17), together, tell us that any pseudo outer vertex might as well be left
                       pseudo during the algorithm.
                          Therefore a pseudo inner vertex should be retained until a planted Hungarian
                       tree JH is obtained. If no inner vertices of JH are pseudo, then (4.17) is applic-
                       able. Otherwise, at this point, a pseudo inner vertex should be expanded
                       according to (7.2).

                          7.4. One of the main operations of the algorithm is described in (4.3). That
                       is back-tracing along paths in a tree already constructed, either to obtain an
                       augmentation as in (4.4) or to delineate a new blossom as in (4.5). The back-
                       tracing takes place in an alternating tree only because blossoms have been
                       shrunk to pseudovertices. A pseudovertex may be compounded from many
                       earlier blossom shrinkings and may thus encompass a complicated subgraph
                       of G. After shrinking, back-tracing entirely bypasses the internal structure of
                       a pseudovertex.
                          A possible alternative to actually shrinking is some method for tracing
                       through the internal structure of a pseudovertex. Witzgall and Zahn (9) have
                       designed a variation of the algorithm which does that. Their result is attractive
                       and deceptively non-trivial.

                                                                           REFERENCES

                       1. C. Berge, Two theorems in graph theory, Proc. Natl. Acad. Sci. U.S., 43 (1957), 842-4.
                       2.         The theory of graphs and its applications (London, 1962).
                       3. J. Edmonds, Covers and packings in a family of sets, Bull. Amer. Math. S o c , 68 (1962),
                              494-9.




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---

                                                 PATHS, TREES, AND FLOWERS                            467

           4.          Maximum matching and a polyhedron with (0, 1) vertices, appearing in J. Res. Natl.
                  Bureau Standards 69B (1965).
           5c L. R. Ford, Jr. and D. R. Fulkerson, Flows in networks (Princeton, 1962).
           6. A. J. Hoffman, Some recent applications of the theory of linear inequalities to extremal com-
                  binatorial analysis, Proc. Symp. on Appl. Math., 10 (1960), 113-27.
           7. R. Z. Norman and M. O. Rabin, An algorithm for a minimum cover of a graph, Proc. Amer.
                  Math. S o c , 10 (1959), 315-19.
           8. W. T. Tutte, The factorization of linear graphs, J. London Math. S o c , 22 (1947), 107-11.
           9. C. Witzgal.l and C. T. Zahn, Jr., Modification of Edmonds} algorithm for maximum matching
                  of graphs, appearing in J. Res. Natl. Bureau Standards 69B (1965).

           National Bureau of Standards and
           Princeton University




https://doi.org/10.4153/CJM-1965-045-4 Published online by Cambridge University Press


--- PAGE BREAK ---


