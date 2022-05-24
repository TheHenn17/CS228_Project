minNum = 100000
valid_auth = 0
with open("uhh.txt", "r") as f:
    small_count = 0
    count = 0
    for line in f.readlines():
        data=line.split()
        count += int(data[0][:-2])
        small_count += int(data[1][:-2])
        if(int(data[0][:-2]) != 0):
            valid_auth += 1
        if (minNum > int(data[1][:-2]) and int(data[1][:-2]) != 0):
            minNum = int(data[1][:-2])

print(count, small_count, 30*valid_auth)