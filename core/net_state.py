class NetState:
    def __init__(self, distance_f, name):
        self.d = distance_f
        self.config = None
        self.net = None
        self.calculated = None
        self.m = None
        self.n = None
        self.name = name
