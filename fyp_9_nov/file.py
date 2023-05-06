import os

txt = os.listdir()

counter = 0
flag = False
for fll in txt:
    if "txt" in fll and "class" not in fll:
        counter +=1
        print(fll)
        f = open(fll, "r")
        # print(f.readline())

        for x in f:
            print(x[0])

            if int(x[0]) > 4:
                flag = True
                break
    if flag:
        break

print(counter)