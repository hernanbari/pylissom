=================
The Visual Cortex
=================

.. role:: raw-latex(raw)
   :format: latex
..

*Brief review of the visual cortex and the visual maps in the cortex.*

Human visual system
===================

During visual perception, light entering the eye is detected by the
retina, an array of photoreceptors and related cells on the inside of
the rear surface of the eye. The cells in the retina encode the light
levels at a given location as patterns of electrical activity in neurons
called ganglion cells. Output from the ganglion cells travels through
neural connections to the lateral geniculate nucleus of the thalamus
(LGN). From the LGN, the signals continue to the primary visual cortex
(V1). V1 is the first cortical site of visual processing; the previous
areas are termed subcortical. The output from V1 goes on to many
different higher cortical areas, including areas that underlie object
and face processing.
   
.. figure:: images/visual_pathways.png
    :align: center
    :alt: alternate text
    :figclass: align-center

**Human visual pathways** (top view). Diagram of the main feedforward pathways in the human visual system.
    
Lateral Geniculate Nucleus (LGN)
--------------------------------

The Lateral Geniculate Nucleus of the thalamus is the first step of
visual processing. The LGN’s neurons do a process akin to edge detection
between bright and dark areas. Some neurons at the LGN prefer bright
over dark, or vice versa, being called ON or OFF cells.


.. figure:: images/oncell.png
   :align: center
   :alt: alternate text   
   
   **ON cell in retina or LGN**
   
.. figure:: images/offcell.png
   :align: center
   :alt: alternate text   
   
   **OFF cell in retina or LGN**

.. _s:v1:

Primary Visual Cortex (V1)
--------------------------

The V1 is the first cortical site of visual processing. Input from the
thalamus goes through afferent connections to V1, and the output goes on
to many different higher cortical areas, including areas that underlie
object and face processing. The neurons form local connections within V1
(long-range lateral connections) or connect to higher visual processing
areas.

The cells themselves can prefer several features, for example,
preferring one eye or the other, the orientation of the stimulus and its
direction of movement, color combinations(such as red/green or
blue/yellow borders), disparity (relative positions on the two retinas),
etc.

.. figure:: images/V1Cell.png
   :align: center
   :alt: alternate text   
   
   **Receptive field in V1.** Starting in V1, most cells in
   primates have orientation-selective RFs. The V1 RFs can be classified
   into a few basic spatial types. The figure shows a two-lobe
   arrangement, favoring a :math:`45^\circ` edge with dark in the upper
   left and light in the lower right
   

At a given location on the cortical sheet, the neurons in a vertical
section through the cortex respond most strongly to the same eye of
origin, stimulus orientation, spatial frequency, and direction of
movement. It is customary to refer to such a section as a column
:raw-latex:`\cite{Gilbert89}`. Nearby columns generally have similar,
but not identical, preferences; slightly more distant columns have more
dissimilar preferences. Preferences repeat at regular intervals
(approximately 1–2 mm) in every direction, which ensures that each type
of preference is represented across the retina. This arrangement of
preferences forms a smoothly varying map for each dimension.

Visual maps
===========

The term *visual map* refers to the existence of a non-random
relationship between the positions of neurons in the visual centers of
the brain (e.g. in the visual cortex) and the values of one or more of
the receptive field properties of those neurons
:raw-latex:`\cite{Swindale:2008}`. The term is usually qualified by
reference to the property concerned. For example, stimulus orientation
is represented across the cortex in an orientation map of the retinal
input. In an orientation map, each location on the retina is mapped to a
region on the map, with each possible orientation at that retinal
location represented by different but nearby orientation-selective
cells. The figure below displays an example orientation map from monkey
cortex.

.. figure:: images/Visual_map.jpg
   :align: center
   :alt: alternate text 

   **Orientation map in the macaque.** Orientation preference map
   observed in area V1 of macaque monkey. Each neuron is colored
   according to the orientation it prefers, using the color key on the
   right. Nearby neurons in the map generally prefer similar
   orientations, forming groups of the same color called iso-orientation
   patches. Scale bar = 1 mm.
   
   
:math:`\alpha > \beta`
   
:math:`n_{\mathrm{offset}} = \sum_{k=0}^{N-1} s_k n_k`
   
