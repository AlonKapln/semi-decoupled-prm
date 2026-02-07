from discopygal.bindings import Point_2, Segment_2
if __name__ == "__main__":

    p1 = Point_2(0, 0)
    p2 = Point_2(1, 1)
    s1 = Segment_2(p1, p2)

    print(s1)