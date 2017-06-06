from pprint import pprint


class FileReader:
    def readfile(self, filePath):
        f = open("../sonnets/"+filePath)
        next = f.readline()
        result = []
        entry = []
        while next != "":
            next = f.readline()
            if next == "\n":
                result.append(entry)
                entry = []
            else:
                entry.append(next)
        return result

if __name__ == "__main__":
    filereader = FileReader()
    pprint(filereader.readfile("misc/misc.txt"))
    pprint(len(filereader.readfile("misc/misc.txt")))
