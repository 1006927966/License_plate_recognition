with open("/code/wujilong/data/车牌/车牌测试/licenseplate_test_cut.txt", "r") as f:
    lines = f.readlines()
count = 0
sum = 0
for line in lines:
    line = line.strip()
    factors = line.split(",")
    gt = factors[0].split("_")[0]
    if len(gt)!=7:
        continue
    sum+= 1
    pre = factors[1]
    if gt==pre:
        count += 1
print("acc is : {}".format(count/sum))