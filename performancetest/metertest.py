import poetrytools

input = """freshborneberslifeguardianyou stead   prospectingbeesbeesrayedhathlowly
 john     bondsesyfice numb 
gaveshookingnathunerlylovedhomageni
facethorerectryserenely fenhotter'dqueencypiteousdrift vailwordedchildreedsmentfree
murs—thewearial none ingsuresshornless graves 
"""

poem = poetrytools.tokenize(input)

print(poetrytools.rhymes("see", "sea"))

print(poetrytools.guess_metre(poem))
print(poetrytools.rhyme_scheme(poem))