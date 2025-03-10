
#
#

class SmoothData:
    def __init__(self, df):
        self.df = df

    def execute(self,labelSmooth,label):
        self.df[labelSmooth] = self.df[label].rolling(window=20, center=True).mean()

