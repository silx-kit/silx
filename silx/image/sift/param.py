class Enum(dict):
    """
    Simple class half way between a dict and a class, behaving as an enum
    """
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError

par = Enum(OctaveMax=100000,
            DoubleImSize=0,
            order=3,
            InitSigma=1.6,
            BorderDist=5,
            Scales=3,
            PeakThresh=255.0 * 0.04 / 3.0,
            EdgeThresh=0.06,
            EdgeThresh1=0.08,
#To detect an edge response, we require the ratio of smallest
#to largest principle curvatures of the DOG function
#(eigenvalues of the Hessian) to be below a threshold.  For
#efficiency, we use Harris' idea of requiring the determinant to
#be above par.EdgeThresh times the squared trace, as for eigenvalues
#A and B, det = AB, trace = A+B.  So if A = 10B, then det = 10B**2,
#and trace**2 = (11B)**2 = 121B**2, so par.EdgeThresh = 10/121 =
#0.08 to require ratio of eigenvalues less than 10.
            OriBins=36,
            OriSigma=1.5,
            OriHistThresh=0.8,
            MaxIndexVal=0.2,
            MagFactor=3,
            IndexSigma=1.0,
            IgnoreGradSign=0,
            MatchRatio=0.73,
            MatchXradius=1000000.0,
            MatchYradius=1000000.0,
            noncorrectlylocalized=0)


