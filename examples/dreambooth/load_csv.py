import csv
import json
import shutil

prefix = 'C:/Users/74590/.cache/huggingface/hub/datasets--nlphuji--flickr30k/blobs/flickr30k-images/'
class Loader:

    def __init__(self):
        return
    def load_csv(self,inputfile = "",outputfile = ""):
        with open('flickr30k/flickr_annotations_30k.csv', newline='') as csvfile:
            reader = csv.reader(
                csvfile, delimiter=",", quotechar='"'
            )
            count = 0
            with open("filter_2_people.csv", 'w', newline='') as writefile:
                writer = csv.writer(writefile)
                for row in reader:
                    if count == 0:
                        writer.writerow(row)
                        count += 1

                        continue
                    l = json.loads(row[0])
                    for caption in list(l):
                        if 'two' in caption.lower():
                            writer.writerow(row)
                            self.copyimg(row[3],"train_image_two_people/")
                            break

        return
        # def readcsv(self,inputfile,outputfolder):
        #     with open(inputfile):
    def copyimg(self,filename,folder):
        shutil.copy(prefix+filename,folder+filename)
c = Loader()
c.load_csv()