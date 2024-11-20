class AngleProcessor:
    def __init__(self, data):
        self.data = data

    def calcAngle(self, A, B, C):
        try:
            BA = [A[0] - B[0], A[1] - B[1], A[2] - B[2]]
            BC = [C[0] - B[0], C[1] - B[1], C[2] - B[2]]

            dot_product = BA[0] * BC[0] + BA[1] * BC[1] + BA[2] * BC[2]
            magnitude_BA = math.sqrt(BA[0] ** 2 + BA[1] ** 2 + BA[2] ** 2)
            magnitude_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2 + BC[2] ** 2)

            if magnitude_BA == 0 or magnitude_BC == 0:
                return 180

            cos_theta = dot_product / (magnitude_BA * magnitude_BC)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle = math.acos(cos_theta)

            return math.degrees(angle)
        except Exception as e:
            print("Error calculating angle:", e)
            return 180

    def getAngle(self):
        data_angle = [[0] * 15 for _ in range(len(self.data))]

        for frame in range(len(self.data)):
            data_angle[frame][0] = self.calcAngle(self.data[frame][0], self.data[frame][1], self.data[frame][2])
            data_angle[frame][1] = self.calcAngle(self.data[frame][1], self.data[frame][2], self.data[frame][3])
            data_angle[frame][2] = self.calcAngle(self.data[frame][0], self.data[frame][4], self.data[frame][5])
            data_angle[frame][3] = self.calcAngle(self.data[frame][4], self.data[frame][5], self.data[frame][6])
            data_angle[frame][4] = self.calcAngle(self.data[frame][1], self.data[frame][0], self.data[frame][7])
            data_angle[frame][5] = self.calcAngle(self.data[frame][4], self.data[frame][0], self.data[frame][7])
            data_angle[frame][6] = self.calcAngle(self.data[frame][0], self.data[frame][7], self.data[frame][8])
            data_angle[frame][7] = self.calcAngle(self.data[frame][7], self.data[frame][8], self.data[frame][9])
            data_angle[frame][8] = self.calcAngle(self.data[frame][8], self.data[frame][9], self.data[frame][10])
            data_angle[frame][9] = self.calcAngle(self.data[frame][10], self.data[frame][9], self.data[frame][11])
            data_angle[frame][10] = self.calcAngle(self.data[frame][9], self.data[frame][11], self.data[frame][12])
            data_angle[frame][11] = self.calcAngle(self.data[frame][1], self.data[frame][12], self.data[frame][13])
            data_angle[frame][12] = self.calcAngle(self.data[frame][10], self.data[frame][9], self.data[frame][14])
            data_angle[frame][13] = self.calcAngle(self.data[frame][9], self.data[frame][14], self.data[frame][15])
            data_angle[frame][14] = self.calcAngle(self.data[frame][14], self.data[frame][15], self.data[frame][16])

        return np.array(data_angle)