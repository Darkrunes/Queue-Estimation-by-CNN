

# When we find a new point thats no in the bounding box of an old point after interpolation
# We create one of these
# This tracks the original bounding box, the current point of the person (updated every optical flow loop)
# Tracks the frame in which they were detected, the confidence, and when they left the frame
class PersonStats:
    def __init__(self, detection_frame_num, bound_box_sizes, detected_points, confidence, p0index, id):
        self.id = id
        self.initial_detection = detection_frame_num
        self.bounding_box_size = bound_box_sizes  # in following way, [x-dist-left, x-dist-right, y-disttop, y-distbot]
        self.currentPoints = [detected_points]
        self.confidence = confidence
        self.p0_indicies = [p0index]
        self.confidence_box_index = 0
        self.last_frame_num = 1000000

    # Checks if a given point is one of those currently saved points for a person
    def point_part_of_person(self, point):
        for curr_point in self.currentPoints:
            if curr_point == point:
                return True
        return False

    # After optical flow iteration we have to update current points to the new points
    def update_points(self, new_points):
        self.currentPoints = new_points

    def update_indicies(self, new_indexs):
        self.p0_indicies = new_indexs

    def add_new_point(self, new_point, confidence, p0index):
        self.currentPoints.append(new_point)
        self.p0_indicies.append(p0index)

        # if we find a higher confidence box that we want to add
        if confidence > self.confidence:
            self.confidence = confidence
