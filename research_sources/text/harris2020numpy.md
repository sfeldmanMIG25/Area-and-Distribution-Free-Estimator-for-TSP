# Array Programming with {NumPy}

> Citation key: `harris2020numpy`  
> DOI: 10.1038/s41586-020-2649-2  
> Download URL: https://arxiv.org/pdf/2006.10256  
> SHA-256: `c6705a2bad0a8b56e992009d1588dd6c7c2c4d657e253a11811dcee0315be791`  

---

                                                                       Array Programming with NumPy

                                             Charles R. Harris1 , K. Jarrod Millman2,3,4,* , Stéfan J. van der Walt5,2,4,* , Ralf
                                         Gommers6,* , Pauli Virtanen7 , David Cournapeau8 , Eric Wieser9 , Julian Taylor10 ,
arXiv:2006.10256v1 [cs.MS] 18 Jun 2020




                                          Sebastian Berg4 , Nathaniel J. Smith11 , Robert Kern12 , Matti Picus4 , Stephan
                                          Hoyer13 , Marten H. van Kerkwijk14 , Matthew Brett2,15 , Allan Haldane16 , Jaime
                                                Fernández del Rı́o17 , Mark Wiebe18,19 , Pearu Peterson6,20,21 , Pierre
                                          Gérard-Marchant22,23 , Kevin Sheppard24 , Tyler Reddy25 , Warren Weckesser4 ,
                                                 Hameer Abbasi6 , Christoph Gohlke26 , and Travis E. Oliphant6
                                                                                  1 Independent Researcher, Logan, Utah, USA
                                                                 2 Brain Imaging Center, University of California, Berkeley, Berkeley, CA, USA
                                                                 3 Division of Biostatistics, University of California, Berkeley, Berkeley, CA, USA
                                                           4 Berkeley Institute for Data Science, University of California, Berkeley, Berkeley, CA, USA
                                                                  5 Applied Mathematics, Stellenbosch University, Stellenbosch, South Africa
                                                                                        6 Quansight LLC, Austin, TX, USA
                                                         7 Department of Physics and Nanoscience Center, University of Jyväskylä, Jyväskylä, Finland
                                                                                            8 Mercari JP, Tokyo, Japan
                                                                    9 Department of Engineering, University of Cambridge, Cambridge, UK
                                                                                10 Independent Researcher, Karlsruhe, Germany
                                                                                 11 Independent Researcher, Berkeley, CA, USA
                                                                                       12 Enthought, Inc., Austin, TX, USA
                                                                                 13 Google Research, Mountain View, CA, USA
                                                            14 Department of Astronomy & Astrophysics, University of Toronto, Toronto, ON, Canada
                                                                15 School of Psychology, University of Birmingham, Edgbaston, Birmigham, UK
                                                                      16 Department of Physics, Temple University, Philadelphia, PA, USA
                                                                                          17 Google, Zurich, Switzerland
                                                     18 Department of Physics and Astronomy, The University of British Columbia, Vancouver, BC, Canada
                                                                                      19 Amazon, Seattle, Washington, USA
                                                                                   20 Independent Researcher, Saue, Estonia
                                         21 Department of Mechanics and Applied Mathematics, Institute of Cybernetics at Tallinn Technical University, Tallinn, Estonia
                                                          22 Department of Biological and Agricultural Engineering, University of Georgia, Athens, GA
                                                                                       23 France-IX Services, Paris, France
                                                                         24 Department of Economics, University of Oxford, Oxford, UK
                                                                       25 CCS-7, Los Alamos National Laboratory, Los Alamos, NM, USA
                                         26 Laboratory for Fluorescence Dynamics, Biomedical Engineering Department, University of California, Irvine, Irvine, CA, USA
                                                                  * millman@berkeley.edu, stefanv@berkeley.edu, ralf.gommers@gmail.com




                                                                                               June 19, 2020




                                                                                                         1


--- PAGE BREAK ---

                                                     Abstract
        Array programming provides a powerful, compact, expressive syntax for accessing, manipulating, and
    operating on data in vectors, matrices, and higher-dimensional arrays [1]. NumPy is the primary array
    programming library for the Python language [2, 3, 4, 5]. It plays an essential role in research analysis
    pipelines in fields as diverse as physics, chemistry, astronomy, geoscience, biology, psychology, material
    science, engineering, finance, and economics. For example, in astronomy, NumPy was an important part of
    the software stack used in the discovery of gravitational waves [6] and the first imaging of a black hole [7].
    Here we show how a few fundamental array concepts lead to a simple and powerful programming paradigm
    for organizing, exploring, and analyzing scientific data. NumPy is the foundation upon which the entire
    scientific Python universe is constructed. It is so pervasive that several projects, targeting audiences with
    specialized needs, have developed their own NumPy-like interfaces and array objects. Because of its central
    position in the ecosystem, NumPy increasingly plays the role of an interoperability layer between these
    new array computation libraries.




   Two Python array packages existed before NumPy.          Python.
The Numeric package began in the mid-1990s and                 NumPy operates on in-memory arrays using the
provided an array object and array-aware functions in       CPU. To utilize modern, specialized storage and
Python, written in C, and linking to standard fast im-      hardware, there has been a recent proliferation of
plementations of linear algebra [8, 9]. One of its ear-     Python array packages. Unlike with the Numarray
liest uses was to steer C++ applications for inertial       and Numeric divide, it is now much harder for these
confinement fusion research at Lawrence Livermore           new libraries to fracture the user community—given
National Laboratory [10]. To handle large astro-            how much work already builds on top of NumPy.
nomical images coming from the Hubble Space Tele-           However, to provide the ecosystem with access to
scope, a reimplementation of Numeric, called Numar-         new and exploratory technologies, NumPy is tran-
ray, added support for structured arrays, flexible in-      sitioning into a central coordinating mechanism that
dexing, memory mapping, byte-order variants, more           specifies a well-defined array programming API and
efficient memory use, flexible IEEE error handling ca-      dispatches it, as appropriate, to specialized array im-
pabilities, and better type casting rules [11]. While       plementations.
Numarray was highly compatible with Numeric, the
two packages had enough differences that it divided         NumPy arrays
the community, until 2005, when NumPy emerged as
a “best of both worlds” unification [12]—combining          The NumPy array is a data structure that efficiently
Numarray’s features with Numeric’s performance on           stores and accesses multidimensional arrays [18], also
small arrays and its rich C Application Programming         known as tensors, and enables a wide variety of scien-
Interface (API).                                            tific computation. It consists of a pointer to memory,
   Now, fifteen years later, NumPy underpins almost         along with metadata used to interpret the data stored
every Python library that does scientific or numeri-        there, notably data type, shape, and strides (Fig. 1a).
cal computation including SciPy [13], Matplotlib [14],         The data type describes the nature of elements
pandas [15], scikit-learn [16], and scikit-image [17].      stored in an array. An array has a single data type,
It is a community-developed, open-source library,           and each array element occupies the same number
which provides a multidimensional Python array ob-          of bytes in memory. Examples of data types include
ject along with array-aware functions that operate on       real and complex numbers (of lower and higher pre-
it. Because of its inherent simplicity, the NumPy ar-       cision), strings, timestamps, and pointers to Python
ray is the de facto exchange format for array data in       objects.


--- PAGE BREAK ---

Fig. 1: The NumPy array incorporates several fundamental array concepts. a, The NumPy
array data structure and its associated metadata fields. b, Indexing an array with slices and steps. These
operations return a view of the original data. c, Indexing an array with masks, scalar coordinates, or other
arrays, so that it returns a copy of the original data. In the bottom example, an array is indexed with other
arrays; this broadcasts the indexing arguments before performing the lookup. d, Vectorization efficiently
applies operations to groups of elements. e, Broadcasting in the multiplication of two-dimensional arrays. f,
Reduction operations act along one or more axes. In this example, an array is summed along select axes to
produce a vector, or along two axes consecutively to produce a scalar. g, Example NumPy code, illustrating
some of these concepts.


   The shape of an array determines the number of           array are therefore (24, 8). NumPy can store arrays
elements along each axis, and the number of axes is         in either C or Fortran memory order, iterating first
the array’s dimensionality. For example, a vector of        over either rows or columns. This allows external li-
numbers can be stored as a one-dimensional array            braries written in those languages to access NumPy
of shape N , while color videos are four-dimensional        array data in memory directly.
arrays of shape (T, M, N, 3).                                  Users interact with NumPy arrays using indexing
   Strides are necessary to interpret computer mem-         (to access subarrays or individual elements), opera-
ory, which stores elements linearly, as multidimen-         tors (e.g., +, −, × for vectorized operations and @ for
sional arrays. It describes the number of bytes to          matrix multiplication), as well as array-aware func-
move forward in memory to jump from row to row,             tions; together, these provide an easily readable, ex-
column to column, and so forth. Consider, for exam-         pressive, high-level API for array programming, while
ple, a 2-D array of floating-point numbers with shape       NumPy deals with the underlying mechanics of mak-
(4, 3), where each element occupies 8 bytes in mem-         ing operations fast.
ory. To move between consecutive columns, we need              Indexing an array returns single elements, subar-
to jump forward 8 bytes in memory, and to access            rays, or elements that satisfy a specific condition
the next row 3 × 8 = 24 bytes. The strides of that          (Fig. 1b). Arrays can even be indexed using other


                                                        3


--- PAGE BREAK ---

arrays (Fig. 1c). Wherever possible, indexing that
retrieves a subarray returns a view on the original ar-
ray, such that data is shared between the two arrays.
This provides a powerful way to operate on subsets
of array data while limiting memory usage.
   To complement the array syntax, NumPy includes
functions that perform vectorized calculations on ar-
rays, including arithmetic, statistics, and trigonom-
etry (Fig. 1d). Vectorization—operating on whole
arrays rather than their individual elements—is es-
sential to array programming. This means that op-
erations that would take many tens of lines to express
in languages such as C can often be implemented as a
single, clear Python expression. This results in con-
cise code and frees users to focus on the details of
their analysis, while NumPy handles looping over ar-
ray elements near-optimally, taking into considera-
tion, for example, strides, to best utilize the com-
puter’s fast cache memory.
   When performing a vectorized operation (such as            Fig. 2: NumPy is the base of the scientific
addition) on two arrays with the same shape, it is            Python ecosystem. Essential libraries and projects
clear what should happen. Through broadcasting,               that depend on NumPy’s API gain access to new
NumPy allows the dimensions to differ, while still            array implementations that support NumPy’s array
producing results that appeal to intuition. A trivial         protocols (Fig. 3).
example is the addition of a scalar value to an array,
but broadcasting also generalizes to more complex
examples such as scaling each column of an array or           backends such as OpenBLAS [19, 20] or Intel MKL
generating a grid of coordinates. In broadcasting,            optimized for the CPUs at hand.
one or both arrays are virtually duplicated (that is,            Altogether, the combination of a simple in-memory
without copying any data in memory), so that the              array representation, a syntax that closely mimics
shapes of the operands match (Fig. 1d). Broadcast-            mathematics, and a variety of array-aware utility
ing is also applied when an array is indexed using            functions forms a productive and powerfully expres-
arrays of indices (Fig. 1c).                                  sive array programming language.
   Other array-aware functions, such as sum, mean,
and maximum, perform element-by-element reduc-
tions, aggregating results across one, multiple, or all       Scientific Python ecosystem
axes of a single array. For example, summing an n-
dimensional array over d axes results in a (n − d)-           Python is an open-source, general-purpose, inter-
dimensional array (Fig. 1f).                                  preted programming language well-suited to standard
   NumPy also includes array-aware functions for cre-         programming tasks such as cleaning data, interacting
ating, reshaping, concatenating, and padding arrays;          with web resources, and parsing text. Adding fast
searching, sorting, and counting data; and reading            array operations and linear algebra allows scientists
and writing files. It provides extensive support for          to do all their work within a single language—and
generating pseudorandom numbers, includes an as-              one that has the advantage of being famously easy
sortment of probability distributions, and performs           to learn and teach, as witnessed by its adoption as a
accelerated linear algebra, utilizing one of several          primary learning language in many universities.


                                                          4


--- PAGE BREAK ---

   Even though NumPy is not part of Python’s stan-            transformations. Matplotlib is used to visualize data
dard library, it benefits from a good relationship with       and to generate the final image of the black hole.
the Python developers. Over the years, the Python                The interactive environment created by the ar-
language has added new features and special syntax            ray programming foundation along with the sur-
so that NumPy would have a more succinct and eas-             rounding ecosystem of tools—inside of IPython or
ier to read array notation. Since it is not part of the       Jupyter—is ideally suited to exploratory data anal-
standard library, NumPy is able to dictate its own            ysis. Users fluidly inspect, manipulate, and visual-
release policies and development patterns.                    ize their data, and rapidly iterate to refine program-
   SciPy and Matplotlib are tightly coupled with              ming statements. These statements are then stitched
NumPy—in terms of history, development, and use.              together into imperative or functional programs, or
SciPy provides fundamental algorithms for scien-              notebooks containing both computation and narra-
tific computing, including mathematical, scientific,          tive. Scientific computing beyond exploratory work
and engineering routines.        Matplotlib generates         is often done in a text editor or an integrated de-
publication-ready figures and visualizations. The             velopment environment (IDEs) such as Spyder. This
combination of NumPy, SciPy, and Matplotlib, to-              rich and productive environment has made Python
gether with an advanced interactive environment like          popular for scientific research.
IPython [21], or Jupyter [22], provides a solid foun-            To complement this facility for exploratory work
dation for array programming in Python. The sci-              and rapid prototyping, NumPy has developed a cul-
entific Python ecosystem (Fig. 2) builds on top of            ture of employing time-tested software engineering
this foundation to provide several, widely used tech-         practices to improve collaboration and reduce error
nique specific libraries [16, 17, 23], that in turn un-       [31]. This culture is not only adopted by leaders in
derlay numerous domain specific projects [24, 25, 26,         the project but also enthusiastically taught to new-
27, 28, 29]. NumPy, at the base of the ecosystem of           comers. The NumPy team was early in adopting dis-
array-aware libraries, sets documentation standards,          tributed revision control and code review to improve
provides array testing infrastructure, and adds build         collaboration on code, and continuous testing that
support for Fortran and other compilers.                      runs an extensive battery of automated tests for ev-
   Many research groups have designed large, com-             ery proposed change to NumPy. The project also
plex scientific libraries, which add application spe-         has comprehensive, high-quality documentation, in-
cific functionality to the ecosystem. For example,            tegrated with the source code [32, 33, 34].
the eht-imaging library [30] developed by the Event              This culture of using best practices for producing
Horizon Telescope collaboration for radio interfer-           reliable scientific software has been adopted by the
ometry imaging, analysis, and simulation, relies on           ecosystem of libraries that build on NumPy. For ex-
many lower-level components of the scientific Python          ample, in a recent award given by the Royal Astro-
ecosystem. NumPy arrays are used to store and ma-             nomical Society to Astropy, they state:
nipulate numerical data at every step in the process-
ing chain: from raw data through calibration and im-              The Astropy Project has provided hun-
age reconstruction. SciPy supplies tools for general              dreds of junior scientists with experience in
image processing tasks such as filtering and image                professional-standard software development
alignment, while scikit-image, an image processing li-            practices including use of version control,
brary that extends SciPy, provides higher-level func-             unit testing, code review and issue tracking
tionality such as edge filters and Hough transforms.              procedures. This is a vital skill set for mod-
The scipy.optimize module performs mathematical                   ern researchers that is often missing from
optimization. NetworkX [23], a package for complex                formal university education in physics or as-
network analysis, is used to verify image comparison              tronomy.
consistency. Astropy [24, 25] handles standard astro-
nomical file formats and computes time/coordinate             Community members explicitly work to address this


                                                          5


--- PAGE BREAK ---

lack of formal education through courses and work-              [40], and JAX arrays all have the capability to run
shops [35, 36, 37].                                             on CPUs and GPUs, in a distributed fashion, uti-
   The recent rapid growth of data science, machine             lizing lazy evaluation to allow for additional per-
learning, and artificial intelligence has further and           formance optimizations. SciPy and PyData/Sparse
dramatically boosted the usage of scientific Python.            both provide sparse arrays—which typically contain
Examples of its significant application, such as the            few non-zero values and store only those in mem-
eht-imaging library, now exist in almost every disci-           ory for efficiency. In addition, there are projects
pline in the natural and social sciences. These tools           that build on top of NumPy arrays as a data con-
have become the primary software environment in                 tainer and extend its capabilities. Distributed arrays
many fields. NumPy and its ecosystem are commonly               are made possible that way by Dask, and labeled
taught in university courses, boot camps, and sum-              arrays—referring to dimensions of an array by name
mer schools, and are at the focus of community con-             rather than by index for clarity, compare x[:, 1] vs.
ferences and workshops worldwide.                               x.loc[:, ’time’]—by xarray [41].
   NumPy and its API have become truly ubiquitous.                 Such libraries often mimic the NumPy API, be-
                                                                cause it lowers the barrier to entry for newcomers
                                                                and provides the wider community with a stable ar-
Array proliferation and interoperability                        ray programming interface. This, in turn, prevents
                                                                disruptive schisms like the divergence of Numeric and
NumPy provides in-memory, multidimensional, ho-                 Numarray. But exploring new ways of working with
mogeneously typed (i.e., single pointer and strided)            arrays is experimental by nature and, in fact, several
arrays on CPUs. It runs on machines ranging from                promising libraries—such as Theano and Caffe—have
embedded devices to the world’s largest supercom-               already ceased development. And each time that
puters, with performance approaching that of com-               a user decides to try a new technology, they must
piled languages. For most its existence, NumPy ad-              change import statements and ensure that the new
dressed the vast majority of array computation use              library implements all the parts of the NumPy API
cases.                                                          they currently use.
   However, scientific data sets now routinely exceed              Ideally, operating on specialized arrays using
the memory capacity of a single machine and may be              NumPy functions or semantics would simply work,
stored on multiple machines or in the cloud. In addi-           so that users could write code once, and would then
tion, the recent need to accelerate deep learning and           benefit from switching between NumPy arrays, GPU
artificial intelligence applications has led to the emer-       arrays, distributed arrays, and so forth, as appropri-
gence of specialized accelerator hardware, including            ate. To support array operations between external
graphics processing units (GPUs), tensor processing             array objects, NumPy therefore added the capability
units (TPUs), and field-programmable gate arrays                to act as a central coordination mechanism with a
(FPGAs). Due to its in-memory data model, NumPy                 well-specified API (Fig. 2).
is currently unable to utilize such storage and special-           To facilitate this interoperability, NumPy provides
ized hardware directly. However, both distributed               “protocols” (or contracts of operation), that allow for
data and the parallel execution of GPUs, TPUs, and              specialized arrays to be passed to NumPy functions
FPGAs map well to the paradigm of array program-                (Fig. 3). NumPy, in turn, dispatches operations to
ming: a gap, therefore, existed between available               the originating library, as required. Over four hun-
modern hardware architectures and the tools neces-              dred of the most popular NumPy functions are sup-
sary to leverage their computational power.                     ported. The protocols are implemented by widely
   The community’s efforts to fill this gap led to a            used libraries such as Dask, CuPy, xarray, and Py-
proliferation of new array implementations. For ex-             Data/Sparse. Thanks to these developments, users
ample, each deep learning framework created its own             can now, for example, scale their computation from
arrays; PyTorch [38], Tensorflow [39], Apache MXNet             a single machine to distributed systems using Dask.


                                                            6


--- PAGE BREAK ---

Fig. 3: NumPy’s API and array protocols expose new arrays to the ecosystem. In this example,
NumPy’s mean function is called on a Dask array. The call succeeds by dispatching to the appropriate library
implementation (i.e., Dask in this case) and results in a new Dask array. Compare this code to the example
code in Fig. 1g.


The protocols also compose well, allowing users to            tran, to manipulate NumPy arrays and pass them
redeploy NumPy code at scale on distributed, multi-           back to Python. Furthermore, using array protocols,
GPU systems via, for instance, CuPy arrays embed-             it is possible to utilize the full spectrum of special-
ded in Dask arrays. Using NumPy’s high-level API,             ized hardware acceleration with minimal changes to
users can leverage highly parallel code execution on          existing code.
multiple systems with millions of cores, all with min-          NumPy was initially developed by students, fac-
imal code changes [42].                                       ulty, and researchers to provide an advanced, open-
   These array protocols are now a key feature of             source array programming library for Python, which
NumPy, and are expected to only increase in impor-            was free to use and unencumbered by license servers,
tance. As with the rest of NumPy, we iteratively re-          dongles, and the like. There was a sense of build-
fine and add protocol designs to improve utility and          ing something consequential together, for the benefit
simplify adoption.                                            of many others. Participating in such an endeavor,
                                                              within a welcoming community of like-minded indi-
                                                              viduals, held a powerful attraction for many early
Discussion                                                    contributors.
NumPy combines the expressive power of array pro-                These user-developers frequently had to write code
gramming, the performance of C, and the readability,          from scratch to solve their own or their colleagues’
usability, and versatility of Python in a mature, well-       problems—often in low-level languages that precede
tested, well-documented, and community-developed              Python, like Fortran [46] and C. To them, the advan-
library. Libraries in the scientific Python ecosystem         tages of an interactive, high-level array library were
provide fast implementations of most important al-            evident. The design of this new tool was informed
gorithms. Where extreme optimization is warranted,            by other powerful interactive programming languages
compiled languages such as Cython [43], Numba [44],           for scientific computing such as Basis [47], Yorick [48],
and Pythran [45], that extend Python and transpar-            R [49], and APL [50], as well as commercial languages
ently accelerate bottlenecks, can be used. Because of         and environments like IDL and MATLAB.
NumPy’s simple memory model, it is easy to write                What began as an attempt to add an array object
low-level, hand-optimized code, usually in C or For-          to Python became the foundation of a vibrant ecosys-


                                                          7


--- PAGE BREAK ---

tem of tools. Now, a large amount of scientific work         quire sustained funding from government, academia,
depends on NumPy being correct, fast, and stable.            and industry. But, importantly, it will also need a
It is no longer a small community project, but is core       new generation of graduate students and other de-
scientific infrastructure.                                   velopers to engage, to build a NumPy that meets the
   The developer culture has matured: while initial          needs of the next decade of data science.
development was highly informal, NumPy now has
a roadmap and a process for proposing and dis-               References
cussing large changes. The project has formal gov-
ernance structures and is fiscally sponsored by Num-          [1] K. E. Iverson, “Notation as a tool of thought,”
FOCUS, a nonprofit that promotes open practices                   Communications of the ACM, vol. 23, p. 444465,
in research, data, and scientific computing. Over                 Aug. 1980.
the past few years, the project attracted its first
funded development, sponsored by the Moore and                [2] P. F. Dubois, “Python: Batteries included,”
Sloan Foundations, and received an award as part of               Computing in Science & Engineering, vol. 9,
the Chan Zuckerberg Initiative’s Essentials of Open               no. 3, pp. 7–9, 2007.
Source Software program. With this funding, the               [3] T. E. Oliphant, “Python for scientific com-
project was (and is) able to have sustained focus over            puting,” Computing in Science & Engineering,
multiple months to implement substantial new fea-                 vol. 9, pp. 10–20, May-June 2007.
tures and improvements. That said, it still depends
heavily on contributions made by graduate students            [4] K. J. Millman and M. Aivazis, “Python for sci-
and researchers in their free time.                               entists and engineers,” Computing in Science &
   NumPy is no longer just the foundational array li-             Engineering, vol. 13, no. 2, pp. 9–12, 2011.
brary underlying the scientific Python ecosystem, but
has also become the standard API for tensor compu-            [5] F. Pérez, B. E. Granger, and J. D. Hunter,
tation and a central coordinating mechanism between               “Python: an ecosystem for scientific comput-
array types and technologies in Python. Work contin-              ing,” Computing in Science & Engineering,
ues to expand on and improve these interoperability               vol. 13, no. 2, pp. 13–21, 2011.
features.                                                     [6] B. P. Abbott, R. Abbott, T. Abbott, M. Aber-
   Over the next decade, we will face several chal-               nathy, F. Acernese, K. Ackley, C. Adams,
lenges. New devices will be developed, and existing               T. Adams, P. Addesso, R. Adhikari, et al., “Ob-
specialized hardware will evolve, to meet diminish-               servation of gravitational waves from a binary
ing returns on Moore’s law. There will be more, and               black hole merger,” Physical Review Letters,
a wider variety of, data science practitioners, a sig-            vol. 116, no. 6, p. 061102, 2016.
nificant proportion of whom will be using NumPy.
The scale of scientific data gathering will continue          [7] A. A. Chael, M. D. Johnson, R. Narayan, S. S.
to expand, with the adoption of devices and instru-               Doeleman, J. F. Wardle, and K. L. Bouman,
ments such as light sheet microscopes and the Large               “High-resolution linear polarimetric imaging for
Synoptic Survey Telescope (LSST) [51]. New gen-                   the event horizon telescope,” The Astrophysical
eration languages, interpreters, and compilers, such              Journal, vol. 829, no. 1, p. 11, 2016.
as Rust [52], Julia [53], and LLVM [54], will invent          [8] P. F. Dubois, K. Hinsen, and J. Hugunin, “Nu-
and determine the viability of new concepts and data              merical Python,” Computers in Physics, vol. 10,
structures.                                                       no. 3, pp. 262–267, 1996.
   Through various mechanisms described in this pa-
per, NumPy is poised to embrace such a changing               [9] D. Ascher, P. F. Dubois, K. Hinsen, J. Hugunin,
landscape, and to continue playing a leading role in              and T. E. Oliphant, “An open source project:
interactive scientific computation. To do so will re-             Numerical Python,” 2001.


                                                         8


--- PAGE BREAK ---

[10] T.-Y. Yang, G. Furnish, and P. F. Dubois,             E. Gouillart, T. Yu, and the scikit-image con-
     “Steering object-oriented scientific computa-         tributors, “scikit-image: image processing in
     tions,” in Proceedings of TOOLS USA 97. In-           Python,” PeerJ, vol. 2, p. e453, 2014.
     ternational Conference on Technology of Object
     Oriented Systems and Languages, pp. 112–119, [18] S. van der Walt, S. C. Colbert, and G. Varo-
     IEEE, 1997.                                           quaux, “The NumPy array: a structure for ef-
                                                           ficient numerical computation,” Computing in
[11] P. Greenfield, J. T. Miller, J. Hsu, and R. L.        Science & Engineering, vol. 13, no. 2, pp. 22–
     White, “numarray: A new scientific array pack-        30, 2011.
     age for Python,” PyCon DC, 2003.
                                                      [19] Q. Wang, X. Zhang, Y. Zhang, and Q. Yi,
[12] T. E. Oliphant, Guide to NumPy. Trelgol Pub-          “Augem: automatically generate high perfor-
     lishing USA, 1st ed., 2006.                           mance dense linear algebra kernels on x86 cpus,”
[13] P. Virtanen, R. Gommers, T. E. Oliphant,              in SC’13: Proceedings of the International Con-
     M. Haberland, T. Reddy, D. Cournapeau,                ference on High Performance Computing, Net-
     E. Burovski, P. Peterson, W. Weckesser,               working, Storage and Analysis, pp. 1–12, IEEE,
     J. Bright, S. J. van der Walt, M. Brett, J. Wil-      2013.
    son, K. J. Millman, N. Mayorov, A. R. J. Nel-
                                                    [20] Z. Xianyi, W. Qian, and Z. Yunquan, “Model-
    son, E. Jones, R. Kern, E. Larson, C. J. Carey,
                                                         driven level 3 blas performance optimization
    I. Polat, Y. Feng, E. W. Moore, J. VanderPlas,
                                                         on loongson 3a processor,” in 2012 IEEE 18th
    D. Laxalde, J. Perktold, R. Cimrman, I. Hen-
                                                         International Conference on Parallel and Dis-
    riksen, E. A. Quintero, C. R. Harris, A. M.
                                                         tributed Systems, pp. 684–691, IEEE, 2012.
    Archibald, A. H. Ribeiro, F. Pedregosa, P. van
    Mulbregt, and SciPy 1.0 Contributors, “SciPy [21] F. Pérez and B. E. Granger, “IPython: a system
    1.0—fundamental algorithms for scientific com-       for interactive scientific computing,” Computing
    puting in Python,” Nature Methods, vol. 17,          in Science & Engineering, vol. 9, no. 3, pp. 21–
    pp. 261–272, 2020.                                   29, 2007.
[14] J. D. Hunter, “Matplotlib: A 2D graphics envi- [22] T. Kluyver, B. Ragan-Kelley, F. Pérez,
     ronment,” Computing in Science & Engineering,       B. Granger, M. Bussonnier, J. Frederic, K. Kel-
     vol. 9, no. 3, pp. 90–95, 2007.                     ley, J. Hamrick, J. Grout, S. Corlay, P. Ivanov,
[15] W. McKinney, “Data structures for statistical       D. Avila, S. Abdalla, and C. Willing, “Jupyter
     computing in Python,” in Proceedings of the 9th     Notebooks—a publishing format for repro-
     Python in Science Conference (S. van der Walt       ducible computational workflows,” in Position-
     and J. Millman, eds.), pp. 51 – 56, 2010.           ing and Power in Academic Publishing: Play-
                                                         ers, Agents and Agendas (F. Loizides and
[16] F. Pedregosa, G. Varoquaux, A. Gramfort,            B. Schmidt, eds.), pp. 87–90, IOS Press, 2016.
     V. Michel, B. Thirion, O. Grisel, M. Blondel,
     P. Prettenhofer, R. Weiss, V. Dubourg, J. Van- [23] A. A. Hagberg, D. A. Schult, and P. J. Swart,
     derplas, A. Passos, D. Cournapeau, M. Brucher,      “Exploring network structure, dynamics, and
     M. Perrot, and É. Duchesnay, “Scikit-learn: Ma-    function using NetworkX,” in Proceedings of the
     chine learning in Python,” Journal of Machine       7th Python in Science Conference (G. Varo-
     Learning Research, vol. 12, no. Oct, pp. 2825–      quaux, T. Vaught, and K. J. Millman, eds.),
     2830, 2011.                                         (Pasadena, CA USA), pp. 11–15, 2008.

[17] S. van der Walt, J. L. Schönberger, J. Nunez- [24] Astropy Collaboration, T. P. Robitaille, E. J.
     Iglesias, F. Boulogne, J. D. Warner, N. Yager,      Tollerud, P. Greenfield, M. Droettboom,


                                                    9


--- PAGE BREAK ---

     E. Bray, T. Aldcroft, M. Davis, A. Ginsburg,               Ninan, M. Nöthe, S. Ogaz, S. Oh, J. K. Pare-
     A. M. Price-Whelan, W. E. Kerzendorf, A. Con-              jko, N. Parley, S. Pascual, R. Patil, A. A.
     ley, N. Crighton, K. Barbary, D. Muna, H. Fer-             Patil, A. L. Plunkett, J. X. Prochaska, T. Ras-
     guson, F. Grollier, M. M. Parikh, P. H. Nair,              togi, V. Reddy Janga, J. Sabater, P. Sakurikar,
     H. M. Unther, C. Deil, J. Woillez, S. Con-                 M. Seifert, L. E. Sherbert, H. Sherwood-Taylor,
     seil, R. Kramer, J. E. H. Turner, L. Singer,               A. Y. Shih, J. Sick, M. T. Silbiger, S. Sin-
     R. Fox, B. A. Weaver, V. Zabalza, Z. I. Ed-                ganamalla, L. P. Singer, P. H. Sladen, K. A.
     wards, K. Azalee Bostroem, D. J. Burke, A. R.              Sooley, S. Sornarajah, O. Streicher, P. Teuben,
     Casey, S. M. Crawford, N. Dencheva, J. Ely,                S. W. Thomas, G. R. Tremblay, J. E. H. Turner,
     T. Jenness, K. Labrie, P. L. Lim, F. Pierfed-              V. Terrón, M. H. van Kerkwijk, A. de la Vega,
     erici, A. Pontzen, A. Ptak, B. Refsdal, M. Servil-         L. L. Watkins, B. A. Weaver, J. B. Whit-
     lat, and O. Streicher, “Astropy: A community               more, J. Woillez, V. Zabalza, and A. Contribu-
     Python package for astronomy,” Astronomy &                 tors, “The Astropy Project: Building an Open-
     Astrophysics, vol. 558, p. A33, Oct. 2013.                 science Project and Status of the v2.0 Core Pack-
                                                                age,” The Astronomical Journal, vol. 156, p. 123,
[25] A. M. Price-Whelan, B. M. Sipőcz, H. M.                   Sept. 2018.
     Günther, P. L. Lim, S. M. Crawford, S. Con-
     seil, D. L. Shupe, M. W. Craig, N. Dencheva,           [26] P. J. Cock, T. Antao, J. T. Chang, B. A.
     A. Ginsburg, J. T. VanderPlas, L. D. Bradley,               Chapman, C. J. Cox, A. Dalke, I. Friedberg,
     D. Pérez-Suárez, M. de Val-Borro, P. Pa-                  T. Hamelryck, F. Kauff, B. Wilczynski, and
     per Contributors, T. L. Aldcroft, K. L. Cruz,               M. J. L. de Hoon, “Biopython: freely available
     T. P. Robitaille, E. J. Tollerud, A. Coordina-              Python tools for computational molecular biol-
     tion Committee, C. Ardelean, T. Babej, Y. P.                ogy and bioinformatics,” Bioinformatics, vol. 25,
     Bach, M. Bachetti, A. V. Bakanov, S. P. Bam-                no. 11, pp. 1422–1423, 2009.
     ford, G. Barentsen, P. Barmby, A. Baumbach,
     K. L. Berry, F. Biscani, M. Boquien, K. A.             [27] K. J. Millman and M. Brett, “Analysis of func-
     Bostroem, L. G. Bouma, G. B. Brammer, E. M.                 tional Magnetic Resonance Imaging in Python,”
     Bray, H. Breytenbach, H. Buddelmeijer, D. J.                Computing in Science & Engineering, vol. 9,
     Burke, G. Calderone, J. L. Cano Rodrı́guez,                 no. 3, pp. 52–55, 2007.
     M. Cara, J. V. M. Cardoso, S. Cheedella,
     Y. Copin, L. Corrales, D. Crichton, D. D’Avella,       [28] T. SunPy Community, S. J. Mumford,
     C. Deil, É. Depagne, J. P. Dietrich, A. Donath,            S. Christe, D. Pérez-Suárez, J. Ireland,
     M. Droettboom, N. Earl, T. Erben, S. Fab-                   A. Y. Shih, A. R. Inglis, S. Liedtke, R. J.
     bro, L. A. Ferreira, T. Finethy, R. T. Fox,                 Hewett, F. Mayer, K. Hughitt, N. Freij,
     L. H. Garrison, S. L. J. Gibbons, D. A. Gold-               T. Meszaros, S. M. Bennett, M. Malocha,
     stein, R. Gommers, J. P. Greco, P. Greenfield,              J. Evans, A. Agrawal, A. J. Leonard, T. P.
     A. M. Groener, F. Grollier, A. Hagen, P. Hirst,             Robitaille, B. Mampaey, J. Iván Campos-Rozo,
     D. Homeier, A. J. Horton, G. Hosseinzadeh,                  and M. S. Kirk, “SunPy—Python for solar
     L. Hu, J. S. Hunkeler, Ž. Ivezić, A. Jain, T. Jen-        physics,” Computational Science and Discovery,
     ness, G. Kanarek, S. Kendrew, N. S. Kern, W. E.             vol. 8, p. 014009, Jan. 2015.
     Kerzendorf, A. Khvalko, J. King, D. Kirkby,
     A. M. Kulkarni, A. Kumar, A. Lee, D. Lenz,             [29] J. Hamman, M. Rocklin, and R. Abernathy,
     S. P. Littlefair, Z. Ma, D. M. Macleod, M. Mas-             “Pangeo: A Big-data Ecosystem for Scalable
     tropietro, C. McCully, S. Montagnac, B. M.                  Earth System Science,” in EGU General Assem-
     Morris, M. Mueller, S. J. Mumford, D. Muna,                 bly Conference Abstracts, EGU General Assem-
     N. A. Murphy, S. Nelson, G. H. Nguyen, J. P.                bly Conference Abstracts, p. 12146, Apr 2018.


                                                        10


--- PAGE BREAK ---

[30] A. A. Chael, K. L. Bouman, M. D. Johnson, [38] A. Paszke, S. Gross, F. Massa, A. Lerer,
     R. Narayan, S. S. Doeleman, J. F. Wardle,               J. Bradbury, G. Chanan, T. Killeen, Z. Lin,
     L. L. Blackburn, K. Akiyama, M. Wielgus, C.-k.          N. Gimelshein, L. Antiga, A. Desmaison,
     Chan, et al., “ehtim: Imaging, analysis, and sim-       A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Te-
     ulation software for radio interferometry,” Astro-      jani, S. Chilamkurthy, B. Steiner, L. Fang,
     physics Source Code Library, 2019.                      J. Bai, and S. Chintala, “Pytorch: An imper-
                                                             ative style, high-performance deep learning li-
[31] K. J. Millman and F. Pérez, “Developing open-          brary,” in Advances in Neural Information Pro-
     source scientific practice,” Implementing Repro-        cessing Systems 32 (H. Wallach, H. Larochelle,
     ducible Research. CRC Press, Boca Raton, FL,            A. Beygelzimer, F. d'Alché-Buc, E. Fox, and
     pp. 149–183, 2014.                                      R. Garnett, eds.), pp. 8024–8035, Curran As-
                                                             sociates, Inc., 2019.
[32] S. van der Walt, “The SciPy documentation
     project (technical overview),” in Proceedings of [39] M. Abadi, A. Agarwal, P. Barham, E. Brevdo,
     the 7th Python in Science Conference (SciPy             Z. Chen, C. Citro, G. S. Corrado, A. Davis,
     2008) (G. Varoquaux, T. Vaught, and K. J. Mill-         J. Dean, M. Devin, et al., “Tensorflow:
     man, eds.), pp. 27–28, 2008.                            Large-scale machine learning on heteroge-
                                                             neous distributed systems,” arXiv preprint
[33] J. Harrington, “The SciPy documentation                 arXiv:1603.04467, 2016.
     project,” in Proceedings of the 7th Python in Sci-
                                                        [40] T. Chen, M. Li, Y. Li, M. Lin, N. Wang,
     ence Conference (SciPy 2008) (G. Varoquaux,
                                                             M. Wang, T. Xiao, B. Xu, C. Zhang,
     T. Vaught, and K. J. Millman, eds.), pp. 33–35,
                                                             and Z. Zhang, “Mxnet: A flexible and ef-
     2008.
                                                             ficient machine learning library for hetero-
[34] J. Harrington and D. Goldsmith, “Progress re-           geneous distributed systems,” arXiv preprint
     port: NumPy and SciPy documentation in                  arXiv:1512.01274, 2015.
     2009,” in Proceedings of the 8th Python in Sci- [41] S. Hoyer and J. Hamman, “xarray: N-D labeled
     ence Conference (SciPy 2009) (G. Varoquaux,          arrays and datasets in Python,” Journal of Open
     S. van der Walt, and K. J. Millman, eds.),           Research Software, vol. 5, no. 1, 2017.
     pp. 84–87, 2009.
                                                     [42] P. Entschev, “Distributed multi-GPU comput-
[35] G. Wilson, “Software carpentry: Getting scien-       ing with Dask, CuPy and RAPIDS.” EuroPy-
     tists to write better code by making them more       thon 2019, 2019.
     productive,” Computing in Science & Engineer-
                                                     [43] S. Behnel, R. Bradshaw, C. Citro, L. Dalcin,
     ing, November–December 2006.
                                                          D. S. Seljebotn, and K. Smith, “Cython: The
[36] J. E. Hannay, H. P. Langtangen, C. MacLeod,          best of both worlds,” Computing in Science &
     D. Pfahl, J. Singer, and G. Wilson, “How do          Engineering, vol. 13, no. 2, pp. 31–39, 2011.
     scientists develop and use scientific software?,” [44] S. K. Lam, A. Pitrou, and S. Seibert, “Numba:
     in Proc. 2009 ICSE Workshop on Software En-            A LLVM-based Python JIT compiler,” in Pro-
     gineering for Computational Science and Engi-          ceedings of the Second Workshop on the LLVM
     neering, 2009.                                         Compiler Infrastructure in HPC, LLVM ’15,
                                                            (New York, NY, USA), pp. 7:1–7:6, ACM, 2015.
[37] K. J. Millman, M. Brett, R. Barnowski, and
     J.-B. Poline, “Teaching computational repro- [45] S. Guelton, P. Brunet, M. Amini, A. Merlini,
     ducibility for neuroimaging,” Frontiers in Neu-        X. Corbillon, and A. Raynaud, “Pythran: En-
     roscience, vol. 12, p. 727, 2018.                      abling static optimization of scientific python


                                                    11


--- PAGE BREAK ---

     programs,” Computational Science & Discovery, [55] P. Peterson, “F2PY: a tool for connecting For-
     vol. 8, no. 1, p. 014001, 2015.                        tran and Python programs,” International Jour-
                                                            nal of Computational Science and Engineering,
[46] J. Dongarra, G. H. Golub, E. Grosse, C. Moler,         vol. 4, no. 4, pp. 296–305, 2009.
     and K. Moore, “Netlib and na-net: Building
     a scientific computing community,” IEEE An- [56] The NumPy Project Community, “NumPy
     nals of the History of Computing, vol. 30, no. 2,      project governance,” 2015.
     pp. 30–41, 2008.
                                                       [57] The NumPy Project Community, “NumPy code
[47] P. F. Dubois, “The basis system,” tech. rep.,          of conduct,” 2018.
     Lawrence Livermore National Laboratory, CA
     (USA), 1989. UCRL-MA-118543, Parts I-VI.          [58] D. Holth, “Pep 427 – the wheel binary package
                                                            format 1.0,” 2012.
[48] D. H. Munro and P. F. Dubois, “Using the yorick
                                                       [59] Brett, M. et al, “multibuild,” 2016.
     interpreted language,” Computers in Physics,
     vol. 9, no. 6, pp. 609–615, 1995.                 [60] B. Griffith, P. Virtanen, N. Smith, M. van Kerk-
                                                            wijk, and S. Hoyer, “NEP 13 – a mechanism for
[49] R. Ihaka and R. Gentleman, “R: a language for
                                                            overriding ufuncs,” 2013.
     data analysis and graphics,” Journal of Compu-
     tational and Graphical Statistics, vol. 5, no. 3, [61] S. Hoyer, M. Rocklin, M. van Kerkwijk, H. Ab-
     pp. 299–314, 1996.                                     basi, and E. Wieser, “NEP 18 – a dispatch mech-
                                                            anism for numpy’s high level array functions,”
[50] K. E. Iverson, “A programming language,” in
                                                            2018.
     Proceedings of the May 1-3, 1962, Spring Joint
     Computer Conference, pp. 345–351, 1962.           [62] M. E. O’Neill, “Pcg: A family of simple fast
                                                            space-efficient statistically good algorithms for
[51] T. Jenness, F. Economou, K. Findeisen, F. Her-
                                                            random number generation,” Tech. Rep. HMC-
     nandez, J. Hoblitt, K. S. Krughoff, K. Lim,
                                                            CS-2014-0905, Harvey Mudd College, Clare-
     R. H. Lupton, F. Mueller, W. O’Mullane, et al.,
                                                            mont, CA, Sept. 2014.
     “Lsst data management software development
     practices and tools,” in Software and Cyber- [63] J. K. Salmon, M. A. Moraes, R. O. Dror, and
     infrastructure for Astronomy V, vol. 10707,            D. E. Shaw, “Parallel random numbers: As easy
     p. 1070709, International Society for Optics and       as 1, 2, 3,” in Proceedings of 2011 International
     Photonics, 2018.                                       Conference for High Performance Computing,
                                                            Networking, Storage and Analysis, SC ’11, (New
[52] N. D. Matsakis and F. S. Klock, “The rust lan-         York, NY, USA), pp. 16:1–16:12, ACM, 2011.
     guage,” Ada Letters, vol. 34, pp. 103–104, Oct.
     2014.                                             [64] C. Doty-Humphrey, “Practrand, version 0.94.”

[53] J. Bezanson, A. Edelman, S. Karpinski, and [65] M. Matsumoto and T. Nishimura, “Mersenne
     V. B. Shah, “Julia: A fresh approach to numer-        Twister: A 623-dimensionally equidistributed
     ical computing,” SIAM Review, vol. 59, no. 1,         uniform pseudo-random number generator,”
     pp. 65–98, 2017.                                      ACM Transactions on Modeling and Computer
                                                           Simulation, vol. 8, pp. 3–30, Jan. 1998.
[54] C. Lattner and V. Adve, “LLVM: A compila-
     tion framework for lifelong program analysis and [66] K. Sheppard, B. Duvenhage, P. de Buyl, and
     transformation,” (San Jose, CA, USA), pp. 75–         D. A. Ham, “bashtage/randomgen: Release
     88, Mar 2004.                                         1.16.2,” Apr. 2019.


                                                     12


--- PAGE BREAK ---

[67] G. Marsaglia and W. W. Tsang, “The ziggurat Methods
     method for generating random variables,” Jour-
     nal of Statistical Software, Articles, vol. 5, no. 8, We use Git for version control and GitHub as the
     pp. 1–7, 2000.                                        public hosting service for our official upstream reposi-
                                                           tory (https://github.com/numpy/numpy). We each
[68] D. Lemire, “Fast random integer generation in work in our own copy (or fork) of the project and use
     an interval,” ACM Transactions on Modeling the upstream repository as our integration point. To
     and Computer Simulation, vol. 29, pp. 1–12, Jan get new code into the upstream repository, we use
     2019.                                                 GitHub’s pull request (PR) mechanism. This allows
                                                           us to review code before integrating it as well as to
[69] top500, “Top 10 sites for november 2019,” 2019.
                                                           run a large number of tests on the modified code to
[70] wikichip, “Astra - supercomputers,” 2019.             ensure that the changes do not break expected be-
                                                           havior.
[71] Wikipedia, “Arm architecture,” 2019.                     We also use GitHub’s issue tracking system to col-
                                                           lect and triage problems and proposed improvements.
[72] NumPy Developers, “Numpy roadmap,” 2019.
[73] Dustin Ingram, “Pep 599 – the manylinux2014             Library organization
     platform tag,” 2019.
                                                             Broadly, the NumPy library consists of the follow-
                                                             ing parts: the NumPy array data structure ndarray;
                                                             the so-called universal functions; a set of library
                                                             functions for manipulating arrays and doing scien-
                                                             tific computation; infrastructure libraries for unit
                                                             tests and Python package building; and the program
                                                             f2py for wrapping Fortran code in Python [55]. The
                                                             ndarray and the universal functions are generally
                                                             considered the core of the library. In the following,
                                                             we give a brief summary of these components of the
                                                             library.

                                                             Core. The ndarray data structure and the univer-
                                                             sal functions make up the core of NumPy.
                                                                The ndarray is the data structure at the heart of
                                                             NumPy. The data structure stores regularly strided
                                                             homogeneous data types inside a contiguous block
                                                             memory, allowing for the efficient representation of n-
                                                             dimensional data. More details about the data struc-
                                                             ture are given in “The NumPy array: a structure for
                                                             efficient numerical computation” [18].
                                                                The universal functions, or more concisely, ufuncs,
                                                             are functions written in C that implement efficient
                                                             looping over NumPy arrays. An important feature of
                                                             ufuncs is the built-in implementation of broadcasting.
                                                             For example, the function arctan2(x, y) is a ufunc
                                                             that accepts two values and computes tan−1 (y/x).
                                                             When arrays are passed in as the arguments, the


                                                        13


--- PAGE BREAK ---

ufunc will take care of looping over the dimensions       tion, installation, and packaging of libraries depend-
of the inputs in such a way that if, say, x is a 1-D      ing on NumPy. These can be used, for example, when
array with length 3, and y is a 2-D array with shape      publishing to the PyPI website.
2 × 1, the output will be an array with shape 2 × 3
(Fig. 1c). The ufunc machinery takes care of calling      F2PY. The program f2py is a tool for building
the function with all the appropriate combinations        NumPy-aware Python wrappers of Fortran functions.
of input array elements to complete the output ar-        NumPy itself does not use any Fortran code; F2PY
ray. The elementary arithmetic operations of addi-        is part of NumPy for historical reasons.
tion, multiplication, etc., are implemented as ufuncs,
so that broadcasting also applies to expressions such
as x + y * z.
                                                          Governance
                                                          NumPy adopted an official Governance Document on
Computing libraries. NumPy provides a large li-           October 5, 2015 [56]. Project decisions are usually
brary of functions for array manipulation and scien-      made by consensus of interested contributors. This
tific computing, including functions for: creating, re-   means that, for most decisions, everyone is entrusted
shaping, concatenating, and padding arrays; search-       with veto power. A Steering Council, currently com-
ing, sorting and counting data in arrays; computing       posed of 12 members, facilitates this process and
elementary statistics, such as the mean, median, vari-    oversees daily development of the project by con-
ance, and standard deviation; file I/O; and more.         tributing code and reviewing contributions from the
   A suite of functions for computing the fast Fourier    community.
transform (FFT) and its inverse is provided.                 NumPy’s official Code of Conduct was approved
   NumPy’s linear algebra library includes functions      on September 1, 2018 [57]. In brief, we strive to:
for: solving linear systems of equations; computing       be open; be empathetic, welcoming, friendly, and pa-
various functions of a matrix, including the determi-     tient; be collaborative; be inquisitive; and be careful in
nant, the norm, the inverse, and the pseudo-inverse;      the words that we choose. The Code of Conduct also
computing the Cholesky, eigenvalue, and singular          specifies how breaches can be reported and outlines
value decompositions of a matrix; and more.               the process for responding to such reports.
   The random number generator library in NumPy
provides alternative bit stream generators that pro-      Funding
vide the core function of generating random integers.
                                                        In 2017, NumPy received its first large grants total-
A higher-level generator class that implements an as-
                                                        ing 1.3M USD from the Gordon & Betty Moore and
sortment of probability distributions is provided. It
                                                        the Alfred P. Sloan foundations. Stfan van der Walt
includes the beta, gamma and Weibull distributions,
                                                        is the PI and manages four programmers working on
the univariate and multivariate normal distributions,
                                                        the project. These two grants focus on addressing
and more.
                                                        the technical debt accrued over the years and on set-
                                                        ting in place standards and architecture to encourage
Infrastructure libraries. NumPy provides utili- more sustainable development.
ties for writing tests and for building Python pack-       NumPy received a third grant for 195K USD from
ages.                                                   the Chan Zuckerberg Initiative at the end of 2019
   The testing subpackage provides functions such with Ralf Gommers as the PI. This grant focuses on
as assert allclose(actual, desired) that may better serving NumPy’s large number of beginning
be used in test suites for code that uses NumPy ar- to intermediate level users and on growing the com-
rays.                                                   munity of NumPy contributors. It will also provide
   NumPy provides the subpackage distutils which support to OpenBLAS, on which NumPy depends for
includes functions and classes to facilitate configura- accelerated linear algebra.


                                                      14


--- PAGE BREAK ---

  Finally, since May 2019 the project receives a small
amount annually from Tidelift, which is used to fund                                        250    Maintainer
                                                                                                   Other




                                                                  PRs merged each quarter
things like documentation and website improvements.
                                                                                            200
                                                                                            150
Developers
                                                                                            100
NumPy is currently maintained by a group of 23
contributors with commit rights to the NumPy code                                           50
base. Out of these, 17 maintainers were active in
2019, 4 of whom were paid to work on the project                                             0
                                                                                                 2   3   4   5   6   7   8   9    0
full-time. Additionally, there are a few long term de-                                        201 201 201 201 201 201 201 201 202
velopers who contributed and maintain specific parts
of NumPy, but are not officially maintainers.                  Fig. 4: Number of pull requests merged into
   Over the course of its history, NumPy has attracted         the NumPy master branch for each quarter
PRs by 823 contributors. However, its development              since 2012. The total number of PRs is indicated
relies heavily on a small number of active maintain-           with the lower blue area showing the portion con-
ers, who share more than half of the contributions             tributed by current or previous maintainers.
among themselves.
   At a release cycle of about every half year, the five
recent releases in the years 2018 and 2019 have aver-          to discuss community concerns.
aged about 450 PRs each,1 with each release attract-
ing more than a hundred new contributors. Figure 4
                                                               NumPy enhancement proposals
shows the number of PRs merged into the NumPy
master branch. Although the number of PRs be-                  Given the complexity of the codebase and the massive
ing merged fluctuates, the plot indicates an increased         number of projects depending on it, large changes re-
number of contributions over the past years.                   quire careful planning and substantial work. NumPy
                                                               Enhancement Proposals (NEPs) are modeled after
Community calls                                                Python Enhancement Proposals (PEPs) for “propos-
                                                               ing major new features, for collecting community in-
The massive number of scientific Python packages               put on an issue, and for documenting the design de-
that built on NumPy meant that it had an unusu-                cisions that have gone into Python”2 . Since then
ally high need for stability. So to guide our devel-           there have been 19 proposed NEPS—6 have been im-
opment we formalized the feature proposal process,             plemented, 4 have been accepted and are being im-
and constructed a development roadmap with exten-              plemented, 4 are under consideration, 3 have been
sive input and feedback from the community.                    deferred or superseded, and 2 have been rejected or
   Weekly community calls alternate between triage             withdrawn.
and higher level discussion. The calls not only involve
developers from the community, but provide a venue
for vendors and other external groups to provide in-
                                                             Central role
put. For example, after Intel produced a forked ver- NumPy plays a central role in building and standard-
sion of NumPy, one of their developers joined a call izing much of the scientific Python community infras-
   1 Note that before mid 2011, NumPy development did not    tructure. NumPy’s docstring standard is now widely
happen on github.com. All data provided here is based on the adopted. We are also now using the NEP system as
development which happened through GitHub PRs. In some a way to help coordinate the larger scientific Python
cases contributions by maintainers may not be categorized as
such.                                                            2 https://numpy.org/neps/nep-0000.html




                                                           15


--- PAGE BREAK ---

community. For example, in NEP 29, we recommend,           scale computing.
along with leaders from various other projects, that
all projects across the Scientific Python ecosystem        Array function protocol
adopt a common “time window-based” policy for sup-
port of Python and NumPy versions. This standard           A vast number of projects are built on NumPy; these
will simplify downstream project and release plan-         projects are consumers of the NumPy API. Over the
ning.                                                      last several years, a growing number of projects are
                                                           providers of a NumPy-like API and array objects
                                                           targeting audiences with specialized needs beyond
Wheels build system                                        NumPy’s capabilities. For example, the NumPy API
A Python wheel [58] is a standard file format for dis-     is implemented by several popular tensor computa-
tributing Python libraries. In addition to Python          tion libraries including CuPy3 , JAX4 , and Apache
code, a wheel may include compiled C extensions and        MXNet5 . PyTorch6 and Tensorflow7 provide ten-
other binary data. This is important, because many         sor APIs with NumPy-inspired semantics. It is also
libraries, including NumPy, require a C compiler and       implemented in packages that support sparse arrays
other build tools to build the software from the source    such as scipy.sparse and PyData/Sparse. Another
code, making it difficult for many users to install the    notable example is Dask, a library for parallel com-
software on their own. The introduction of wheels          puting in Python. Dask adopts the NumPy API
to the Python packaging system has made it much            and therefore presents a familiar interface to exist-
easier for users to install precompiled libraries.         ing NumPy users, while adding powerful abilities to
   A GitHub repository containing scripts to build         parallelize and distribute tasks.
NumPy wheels has been configured so that a simple             The multitude of specialized projects creates the
commit to the repository triggers an automated build       difficulty that consumers of these NumPy-like APIs
system that creates NumPy wheels for several com-          write code specific to a single project and do not sup-
puter platforms, including Windows, Mac OSX and            port all of the above array providers. This is a burden
Linux. The wheels are uploaded to a public server          for users relying on the specialized array-like, since a
and made available for anyone to use. This system          tool they need may not work for them. It also cre-
makes it easy for users to install precompiled versions    ates challenges for end-users who need to transition
of NumPy on these platforms.                               from NumPy to a more specialized array. The grow-
   The technology that is used to build the wheels         ing multitude of specialized projects with NumPy-
evolves continually. At the time this paper is being       like APIs threatened to again fracture the scientific
written, a key component is the multibuild suite           Python community.
of tools developed by Matthew Brett and other de-             To address these issues NumPy has the goal of
velopers [59]. Currently, scripts using multibuild         providing the fundamental API for interoperability
are written for the continuous integration platforms       between the various NumPy-like APIs. An earlier
Travis-CI (for Linux and Mac OSX) and Appveyor             step in this direction was the implementation of the
(for Windows).                                               array ufunc protocol in NumPy 1.13, which en-
                                                           abled interoperability for most mathematical func-
                                                           tions [60]. In 2019 this was expanded more generally
Recent technical improvements
                                                             3 https://cupy.chainer.org/
With the recent infusion of funding and a clear pro-    4 https://jax.readthedocs.io/en/latest/jax.numpy.
cess for coordinating with the developer community, html
                                                        5 https://numpy.mxnet.io/
we have been able to tackle a number of important       6 https://pytorch.org/tutorials/beginner/blitz/
large scale changes. We highlight two of those be- tensor_tutorial.html
low, as well as changes made to our testing infras-     7 https://www.tensorflow.org/tutorials/

tructure to support hardware platforms used in large customization/basics


                                                      16


--- PAGE BREAK ---

with the inclusion of the array function pro-                that transforms random bits into variates from other
tocol into NumPy 1.17. These two protocols allow             distributions; and supplies a singleton instance ex-
providers of array objects to be interoperable with          posed in the root of the random module.
the NumPy API: their arrays work correctly with al-             The RandomState object makes a compatibility
most all NumPy functions [61]. For the users rely-           guarantee so that a fixed seed and sequence of func-
ing on specialized array projects it means that even         tion calls produce the same set of values. This guar-
though much code is written specifically for NumPy           antee has slowed progress since improving the under-
arrays and uses the NumPy API as import numpy                lying code requires extending the API with additional
as np, it can nevertheless work for them. For exam-          keyword arguments. This guarantee continues to ap-
ple, here is how a CuPy GPU array can be passed              ply to RandomState.
through NumPy for processing, with all operations               NumPy 1.17 introduced a new API for generating
being dispatched back to CuPy:                               random numbers that use a more flexible structure
  import numpy as np                                         that can be extended by libraries or end-users. The
  import cupy as cp                                          new API is built using components that separate the
                                                             steps required to generate random variates. Pseudo-
  x_gpu = cp . array ([1 , 2 , 3])
                                                             random bits are generated by a bit generator. These
  y = np . sum ( x_gpu ) # Returns a GPU array
                                                             bits are then transformed into variates from complex
   Similarly, user defined functions composed using          distributions by a generator. Finally, seeding is han-
NumPy can now be applied to, e.g., multi-node dis-           dled by an object that produces sequences of high-
tributed Dask arrays:                                        quality initial values.
  import numpy as np                                            Bit generators are simple classes that manage
  import dask . array as da                                  the state of an underlying pseudorandom number
                                                             generator. NumPy ships with four bit generators.
                                                             The default bit generator is a 64-bit implementa-
  def f ( x ) :
      """ Function using NumPy API calls """                 tion of the Permuted Congruential Generator [62]
      y = np . tensordot (x , x . T )                        (PCG64). The three other bit generators are a 64-
      return np . mean ( np . log ( y + 1) )                 bit version of the Philox generator [63] (Philox),
                                                             Chris Doty-Humphrey’s Small Fast Chaotic genera-
  x_local = np . random . random ([10000 , 10000])           tor [64] (SFC64), and the 32-bit Mersenne Twister [65]
       # random local array                                  (MT19937) which has been used in older versions of
  x_distr = da . random . random ([10000 , 10000])           NumPy.9 Bit generators provide functions, exposed
       # random distributed array
                                                             both in Python and C, for generating random integer
  f ( x_local )   # returns a NumPy array                    and floating point numbers.
  f ( x_distr )   # works , returns a Dask array                The Generator consumes one of the bit gener-
                                                             ators and produces variates from complicated dis-
                                                             tributions. Many improved methods for generat-
Random number generation                                     ing random variates from common distributions were
The NumPy random module provides pseudorandom                implemented, including the Ziggurat method for
numbers from a wide range of distributions. In legacy        normal, exponential and gamma variates [67], and
versions of NumPy, simulated random values are pro-          Lemire’s method for bounded random integer gen-
duced by a RandomState object that: handles seeding          eration [68]. The Generator is more similar to the
and state initialization; wraps the core pseudoran-          legacy RandomState, and its API is substantially the
dom number generator based on a Mersenne Twister         9 The randomgen project supplies a wide range of alterna-

implementation8 ; interfaces with the underlying code tive bit generators such as a cryptographic counter-based gen-
                                                             erators (AESCtr) and generators that expose hardware random
  8 to be precise, the standard 32-bit version of MT19937    number generators (RDRAND) [66].


                                                            17


--- PAGE BREAK ---

same. The key differences all relate to state manage-            instantiate a Generator when reproducibility is not
ment, which has been delegated to the bit genera-                needed.
tor. The Generator does not make the same stream                    The final goal of the new API is to improve ex-
guarantee as the RandomState object, and so variates             tensibility. RandomState is a monolithic object that
may differ across versions as improved generation al-            obscures all of the underlying state and functions.
gorithms are introduced.10                                       The component architecture is one part of the ex-
   Finally, a SeedSequence is used to initialize a bit           tensibility improvements. The underlying functions
generator. The seed sequence can be initialized with             (written in C) which transform the output of a bit
no arguments, in which case it reads entropy from a              generator to other distributions are available for use
system-dependent provider, or with a user-provided               in CFFI. This allows the same code to be run in both
seed. The seed sequence then transforms the initial              NumPy and dependent that can consume CFFI, e.g.,
set of entropy into a sequence of high-quality pseu-             Numba. Both the bit generators and the low-level
dorandom integers, which can be used to initialize               functions can also be used in C or Cython code.11
multiple bit generators deterministically. The key
feature of a seed sequence is that it can be used to             Testing on multiple architectures
spawn child SeedSequences to initialize multiple dis-
tinct bit generators. This capability allows a seed At the time of writing the two fastest supercomput-
                                                    ers in the world, Summit and Sierra, both have IBM
sequence to facilitate large distributed applications
where the number of workers required is not known.  POWER9 architectures [69]. In late 2018, Astra, the
The sequences generated from the same initial en-   first ARM-based supercomputer to enter the TOP500
tropy and spawns are fully deterministic to ensure  list, went into production [70]. Furthermore, over
reproducibility.                                    100 billion ARM processors have been produced as
   The three components are combined to construct   of 2017 [71], making it the most widely used instruc-
a complete random number generator.                 tion set architecture in the world.
                                                       Clearly there are motivations for a large scien-
  from numpy . random import (
       Generator ,
                                                    tific computing software library to support POWER
      PCG64 ,                                       and ARM architectures. We’ve extended our con-
       SeedSequence ,                               tinuous integration (CI) testing to include ppc64le
  )                                                 (POWER8 on Travis CI) and ARMv8 (on Shippable
  seq = SeedSequence                                service). We also test with the s390x architecture
      (1030424547444117993331016959)                (IBM Z CPUs on Travis CI) so that we can probe the
  pcg = PCG64 ( seq )                               behavior of our library on a big-endian machine. This
  gen = Generator ( pcg )                           satisfies one of the major components of improved CI
  This approach retains access to the seed sequence testing laid out in a version of our roadmap [72]—
which can then be used to spawn additional genera- specifically, “CI for more exotic platforms.”
tors.                                                  PEP 599 [73] lays out a plan for new Python
                                                    binary wheel distribution support, manylinux2014,
  children = seq . spawn (2)                        that adds support for a number of architectures sup-
  gen_0 = Generator ( PCG64 ( children [0]) )
  gen_1 = Generator ( PCG64 ( children [1]) )
                                                    ported by the CentOS Alternative Architecture Spe-
                                                    cial Interest Group, including ARMv8, ppc64le, as
  While this approach retains complete flexibility, well as s390x. We are thus well-positioned for a fu-
the method np.random.default rng can be used to ture where provision of binaries on these architec-
                                                    tures will be expected for a library at the base of the
   10 Despite the removal of the compatibility guarantee, sim-

ple reproducibility across versions is encouraged, and minor       11 As of 1.18.0, this scenario requires access to the NumPy

changes that do not produce meaningful performance gains or      source. Alternative approaches that avoid this extra step are
fix underlying bug are not generally adopted.                    being explored.


                                                             18


--- PAGE BREAK ---

ecosystem.                                              its value. Others injected new energy and ideas by
                                                        creating experimental array packages.
                                                           K.J.M. and S.J.v.d.W. were funded in part by the
Acknowledgments                                         Gordon and Betty Moore Foundation through Grant
                                                        GBMF3834 and by the Alfred P. Sloan Foundation
We thank Ross Barnowski, Paul Dubois, Michael           through Grant 2013-10-27 to the University of Cali-
Eickenberg, and Perry Greenfield, who suggested text    fornia, Berkeley. S.J.v.d.W., S.B., M.P., and W.W.
and provided helpful feedback on the manuscript.        were funded in part by the Gordon and Betty Moore
   We also thank the many members of the commu-         Foundation through Grant GBMF5447 and by the
nity who provided feedback, submitted bug reports,      Alfred P. Sloan Foundation through Grant G-2017-
made improvements to the documentation, code, or        9960 to the University of California, Berkeley.
website, promoted NumPy’s use in their scientific
fields, and built the vast ecosystem of tools and li-
braries around NumPy. We also gratefully acknowl-       Author Contributions Statement
edge the Numeric and Numarray developers on whose
                                                        K.J.M. and S.J.v.d.W. composed the manuscript
work we built.
                                                        with input from others. S.B., R.G., K.S., W.W.,
   Jim Hugunin wrote Numeric in 1995, while a grad-
                                                        M.B., and T.J.R. contributed text. All authors have
uate student at MIT. Hugunin based his package on
                                                        contributed significant code, documentation, and/or
previous work by Jim Fulton, then working at the US
                                                        expertise to the NumPy project. All authors re-
Geological Survey, with input from many others. Af-
                                                        viewed the manuscript.
ter he graduated, Paul Dubois at the Lawrence Liv-
ermore National Laboratory became the maintainer.
Many people contributed to the project including        Competing Interests
T.E.O. (a co-author of this paper), David Ascher,
Tim Peters, and Konrad Hinsen.                          The authors declare no competing interests.
   In 1998 the Space Telescope Science Institute
started using Python and in 2000 began developing a
new array package called Numarray, written almost
entirely by Jay Todd Miller, starting from a proto-
type developed by Perry Greenfield. Other contrib-
utors included Richard L. White, J. C. Hsu, Jochen
Krupper, and Phil Hodge. The Numeric/Numarray
split divided the community, yet ultimately pushed
progress much further and faster than would other-
wise have been possible.
   Shortly after Numarray development started,
T.E.O. took over maintenance of Numeric. In 2005,
he led the effort and did most of the work to unify
Numeric and Numarray, and produce the first version
of NumPy.
   Eric Jones co-founded (along with T.E.O. and P.P.)
the SciPy community, gave early feedback on array
implementations, and provided funding and travel
support to several community members. Numerous
people contributed to the creation and growth of the
larger SciPy ecosystem, which gives NumPy much of


                                                    19


--- PAGE BREAK ---


