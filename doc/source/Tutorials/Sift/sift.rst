
SIFT image alignment tutorial
=============================

SIFT (Scale-Invariant Feature Transform) is an algorithm developped by
David Lowe in 1999. It is a worldwide reference for image alignment and
object recognition. The robustness of this method enables to detect
features at different scales, angles and illumination of a scene. Silx
provides an implementation of SIFT in OpenCL, meaning that it can run on
Graphics Processing Units and Central Processing Units as well. Interest
points are detected in the image, then data structures called
*descriptors* are built to be characteristic of the scene, so that two
different images of the same scene have similar descriptors. They are
robust to transformations like translation, rotation, rescaling and
illumination change, which make SIFT interesting for image stitching. In
the fist stage, descriptors are computed from the input images. Then,
they are compared to determine the geometric transformation to apply in
order to align the images. This implementation can run on most graphic
cards and CPU, making it usable on many setups. OpenCL processes are
handled from Python with PyOpenCL, a module to access OpenCL parallel
computation API.

This tutuorial explains the three subsequent steps:

-  keypoint extraction
-  Keypoint matching
-  image alignment

All the tutorial has been made using the Jupyter notebook.

.. code:: python

    %pylab inline


.. parsed-literal::

    Populating the interactive namespace from numpy and matplotlib


.. code:: python

    # display test image
    import scipy.misc
    image = scipy.misc.ascent()
    imshow(image, cmap="gray")




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x7fc3a7ca30f0>




.. image:: output_2_1.png


.. code:: python

    #Initialization of the sift object is time consuming: it compiles all the code.
    import os 
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0" #set to 1 to see the compilation going on
    from silx.image import sift
    %time sift_ocl = sift.SiftPlan(template=image, devicetype="CPU") #switch to GPU to test your graphics card


.. parsed-literal::

    CPU times: user 228 ms, sys: 8 ms, total: 236 ms
    Wall time: 225 ms


.. parsed-literal::

    /scisoft/users/jupyter/jupy34/lib/python3.4/site-packages/pyopencl/__init__.py:207: CompilerWarning: Non-empty compiler output encountered. Set the environment variable PYOPENCL_COMPILER_OUTPUT=1 to see more.
      "to see more.", CompilerWarning)


.. code:: python

    print("Time for calculating the keypoints on one image of size %sx%s"%image.shape)
    %time keypoints = sift_ocl.keypoints(image)
    print("Number of keypoints: %s"%len(keypoints))
    print("Keypoint content:")
    print(keypoints.dtype)
    print("x: %.3f \t y: %.3f \t sigma: %.3f \t angle: %.3f" % 
          (keypoints[-1].x,keypoints[-1].y,keypoints[-1].scale,keypoints[-1].angle))
    print("descriptor:")
    print(keypoints[-1].desc)


.. parsed-literal::

    Time for calculating the keypoints on one image of size 512x512
    CPU times: user 1.48 s, sys: 184 ms, total: 1.67 s
    Wall time: 879 ms
    Number of keypoints: 491
    Keypoint content:
    (numpy.record, [('x', '<f4'), ('y', '<f4'), ('scale', '<f4'), ('angle', '<f4'), ('desc', 'u1', (128,))])
    x: 287.611 	 y: 127.560 	 sigma: 47.461 	 angle: 0.503
    descriptor:
    [  0   0   0   0   0   0   0   0  13   0   0   5   3   0   0   2  49  12
       5   0   0   0   5  27   1   3   7   0   0   0   3   4   0   7  13  24
      40   0   0   0  61  11   4  72 127   3   0   8 127  92  34   3   7   0
       2  54  22  48  52   0   0   7  14  18   0  33 111 101 127   6   0   0
      33 127  37  64 127  11   0   4 125 127  91  17   5   0   7  54  20   8
      10  12   9  37  50  39   0   2  14  30 127  97   5   0  53  40   6  25
     119  58  15  54  58  30  17  13   7   7  12  67   0   0   2  21  36  25
       4   1]


.. code:: python

    #Overlay keypoints on the image:
    imshow(image, cmap="gray")
    plot(keypoints[:].x, keypoints[:].y,".")




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x7fc3e403c6a0>]




.. image:: output_5_1.png


.. code:: python

    #Diplaying keypoints by scale:
    hist(keypoints[:].scale, 100)




.. parsed-literal::

    (array([ 113.,  100.,   61.,   47.,   25.,   26.,   14.,   22.,   10.,
               8.,    7.,    5.,    2.,    4.,    2.,    5.,    1.,    4.,
               4.,    3.,    1.,    0.,    0.,    0.,    2.,    3.,    1.,
               1.,    2.,    1.,    1.,    3.,    0.,    1.,    1.,    0.,
               0.,    2.,    1.,    0.,    0.,    0.,    0.,    0.,    0.,
               0.,    0.,    0.,    2.,    0.,    1.,    0.,    1.,    0.,
               0.,    0.,    0.,    1.,    0.,    0.,    0.,    0.,    0.,
               0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
               0.,    0.,    0.,    0.,    1.,    0.,    0.,    0.,    0.,
               0.,    0.,    0.,    1.,    0.,    0.,    0.,    0.,    0.,
               0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    1.]),
     array([  1.69660795,   2.15425151,   2.61189507,   3.06953864,
              3.5271822 ,   3.98482576,   4.44246932,   4.90011289,
              5.35775645,   5.81540001,   6.27304357,   6.73068714,
              7.1883307 ,   7.64597426,   8.10361782,   8.56126139,
              9.01890495,   9.47654851,   9.93419207,  10.39183564,
             10.8494792 ,  11.30712276,  11.76476632,  12.22240989,
             12.68005345,  13.13769701,  13.59534057,  14.05298414,
             14.5106277 ,  14.96827126,  15.42591482,  15.88355839,
             16.34120195,  16.79884551,  17.25648907,  17.71413264,
             18.1717762 ,  18.62941976,  19.08706332,  19.54470689,
             20.00235045,  20.45999401,  20.91763757,  21.37528114,
             21.8329247 ,  22.29056826,  22.74821182,  23.20585539,
             23.66349895,  24.12114251,  24.57878608,  25.03642964,
             25.4940732 ,  25.95171676,  26.40936033,  26.86700389,
             27.32464745,  27.78229101,  28.23993458,  28.69757814,
             29.1552217 ,  29.61286526,  30.07050883,  30.52815239,
             30.98579595,  31.44343951,  31.90108308,  32.35872664,
             32.8163702 ,  33.27401376,  33.73165733,  34.18930089,
             34.64694445,  35.10458801,  35.56223158,  36.01987514,
             36.4775187 ,  36.93516226,  37.39280583,  37.85044939,
             38.30809295,  38.76573651,  39.22338008,  39.68102364,
             40.1386672 ,  40.59631076,  41.05395433,  41.51159789,
             41.96924145,  42.42688501,  42.88452858,  43.34217214,
             43.7998157 ,  44.25745926,  44.71510283,  45.17274639,
             45.63038995,  46.08803352,  46.54567708,  47.00332064,  47.4609642 ]),
     <a list of 100 Patch objects>)




.. image:: output_6_1.png


.. code:: python

    #One can see 2 groups of keypoints: <12 and >12. Let's display them using colors.
    S = 8
    L = 20
    tiny = keypoints[keypoints[:].scale<S]
    small = keypoints[numpy.logical_and(keypoints[:].scale<L,keypoints[:].scale>=S)]
    bigger = keypoints[keypoints[:].scale>=L]
    imshow(image, cmap="gray")
    plot(tiny[:].x, tiny[:].y,".g", label="tiny")
    plot(small[:].x, small[:].y,".b", label="small")
    plot(bigger[:].x, bigger[:].y,".r", label="large")
    legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x7fc3a7d19b70>




.. image:: output_7_1.png


Image matching and alignment
----------------------------

Matching can also be performed on the device (GPU) as every single
keypoint from an image needs to be compared with all keypoints from the
second image.

In this simple example we will simple offset the first image by a few
pixels

.. code:: python

    shifted = numpy.zeros_like(image)
    shifted[5:,8:] = image[:-5, :-8]
    shifted_points = sift_ocl.keypoints(shifted)

.. code:: python

    %time mp = sift.MatchPlan()
    %time match = mp.match(keypoints, shifted_points)
    print("Number of Keypoints with for image 1 : %i, For image 2 : %i, Matching keypoints: %i" % (kp1.size, kp2.size, match.shape[0]))
    



::


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-37-8f97defa84d5> in <module>()
    ----> 1 get_ipython().magic('time mp = sift.MatchPlan()')
          2 get_ipython().magic('time match = mp.match(keypoints, shifted_points)')
          3 print("Number of Keypoints with for image 1 : %i, For image 2 : %i, Matching keypoints: %i" % (kp1.size, kp2.size, match.shape[0]))
          4 


    /scisoft/users/jupyter/jupy34/lib/python3.4/site-packages/IPython/core/interactiveshell.py in magic(self, arg_s)
       2334         magic_name, _, magic_arg_s = arg_s.partition(' ')
       2335         magic_name = magic_name.lstrip(prefilter.ESC_MAGIC)
    -> 2336         return self.run_line_magic(magic_name, magic_arg_s)
       2337 
       2338     #-------------------------------------------------------------------------


    /scisoft/users/jupyter/jupy34/lib/python3.4/site-packages/IPython/core/interactiveshell.py in run_line_magic(self, magic_name, line)
       2255                 kwargs['local_ns'] = sys._getframe(stack_depth).f_locals
       2256             with self.builtin_trap:
    -> 2257                 result = fn(*args,**kwargs)
       2258             return result
       2259 


    <decorator-gen-60> in time(self, line, cell, local_ns)


    /scisoft/users/jupyter/jupy34/lib/python3.4/site-packages/IPython/core/magic.py in <lambda>(f, *a, **k)
        191     # but it's overkill for just that one bit of state.
        192     def magic_deco(arg):
    --> 193         call = lambda f, *a, **k: f(*a, **k)
        194 
        195         if callable(arg):


    /scisoft/users/jupyter/jupy34/lib/python3.4/site-packages/IPython/core/magics/execution.py in time(self, line, cell, local_ns)
       1165         else:
       1166             st = clock2()
    -> 1167             exec(code, glob, local_ns)
       1168             end = clock2()
       1169             out = None


    <timed exec> in <module>()


    /scisoft/users/jupyter/jupy34/lib/python3.4/site-packages/silx/image/sift/match.py in __init__(self, size, devicetype, profile, device, max_workgroup_size, roi, context)
        120             self.queue = pyopencl.CommandQueue(self.ctx)
        121 #        self._calc_workgroups()
    --> 122         self._compile_kernels()
        123         self._allocate_buffers()
        124         self.debug = []


    /scisoft/users/jupyter/jupy34/lib/python3.4/site-packages/silx/image/sift/match.py in _compile_kernels(self)
        186         for kernel in self.kernels:
        187             kernel_file = os.path.join(kernel_directory, kernel + ".cl")
    --> 188             kernel_src = open(kernel_file).read()
        189             wg_size = self.kernels[kernel]
        190             try:


    FileNotFoundError: [Errno 2] No such file or directory: '/scisoft/users/jupyter/jupy34/lib/python3.4/site-packages/silx/image/sift/sift_kernels/matching_gpu.cl'


References
~~~~~~~~~~

-  David G. Lowe, Distinctive image features from scale-invariant
   keypoints, International Journal of Computer Vision, vol. 60, no 2,
   2004, p. 91–110 - "http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf"

