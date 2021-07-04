class EnigmaKey:
    def __init__(self, 
        rotors, 
        indicators, 
        rings, 
        plugboardConnections
    ):
        if rotors == None:
            self.rotors ["I", "II", "III"]
        else:
            self.rotors = rotors

        if indicators == None:
            self.indicators = [0,0,0]
        else:
            self.indicators = indicators

        if rings == None:
            self.rings = [0,0,0]
        else:
            self.rings = rings

        if plugboardConnections ==None:
            self.plugboard = ""
        else:
            self.plugboard = plugboardConnections
            
class Reflector:
    def __init__(self, encoding):
        self.forwardWiring = Reflector.decodeWiring(encoding)
    
    @staticmethod
    def Create(name):
        if name == "B":
            return Reflector("YRUHQSLDPXNGOKMIEBFZCWVJAT")
        elif name == "C":
            return Reflector("FVPJIAOYEDRZXWGCTKUQSBNMHL")
        else:
            return Reflector("ZYXWVUTSRQPONMLKJIHGFEDCBA")

    @staticmethod
    def decodeWiring(encoding):
        charWiring = encoding
        wiring = []
        for i in range(len(charWiring)):
            wiring.append(ord(charWiring[i]) - 65)
        return wiring
    
    def forward(self, c):
        return self.forwardWiring[c]
    
class Plugboard:
    def __init__(self, connections):
        self.wiring = self.decodePlugboard(connections)

    @staticmethod
    def nonAlphaSplit(text):
        res = []
        text = "#"+text
        for i in range(len(text)):
            ch = text[i]
            if ch.isalpha():
                prevCh = text[i-1]
                if prevCh.isalpha():
                    res[-1] += ch
                else:
                    res.append(ch)
        return res

    def forward(self, c):
        return self.wiring[c]
    
    @staticmethod
    def identityPlugboard():
        mapping = []
        for i in range(26):
            mapping.append(i)
        return mapping

    @staticmethod
    def getUnpluggedCharacters(plugboard):
        unpluggedCharacters = set()
        for i in range(26):
            unpluggedCharacters.add(i)
        
        if plugboard== "":
            return unpluggedCharacters
        
        pairings = Plugboard.nonAlphaSplit(plugboard)

        # Validate and create mapping
        for pair in pairings:
            c1 = ord(pair[0]) - 65
            c2 = ord(pair[1]) - 65
            unpluggedCharacters.remove(c1)
            unpluggedCharacters.remove(c2)
        
        return unpluggedCharacters
    
    @staticmethod
    def decodePlugboard(plugboard):
        if (plugboard == None or plugboard== ""):
            return Plugboard.identityPlugboard()
        
        pairings = Plugboard.nonAlphaSplit(plugboard)

        pluggedCharacters = set()
        mapping = Plugboard.identityPlugboard()

        # Validate and create mapping
        for pair in pairings:
            if (len(pair) != 2):
                return Plugboard.identityPlugboard()

            c1 = ord(pair[0]) - 65
            c2 = ord(pair[1]) - 65

            if (c1 in pluggedCharacters or c2 in pluggedCharacters):
                return Plugboard.identityPlugboard()
            
            pluggedCharacters.add(c1)
            pluggedCharacters.add(c2)

            mapping[c1] = c2
            mapping[c2] = c1

        return mapping

class Rotor:   
    def __init__(self, 
        name, 
        encoding,
        rotorPosition,
        notchPosition,
        ringSetting,
        overrideIsAtNotch = False
    ):
        self.name = name
        self.forwardWiring = self.decodeWiring(encoding)
        self.backwardWiring = self.inverseWiring(self.forwardWiring)
        self.rotorPosition = rotorPosition
        self.notchPosition = notchPosition
        self.ringSetting = ringSetting
        self.overrideIsAtNotch = overrideIsAtNotch
    

    @staticmethod
    def Create(name, rotorPosition, ringSetting):
        if name == "I":
            return Rotor("I","EKMFLGDQVZNTOWYHXUSPAIBRCJ", rotorPosition, 16, ringSetting)
        elif name == "II":
            return Rotor("II","AJDKSIRUXBLHWTMCQGZNPYFVOE", rotorPosition, 4, ringSetting)
        elif name == "III":
            return Rotor("III","BDFHJLCPRTXVZNYEIWGAKMUSQO", rotorPosition, 21, ringSetting)
        elif name == "IV":
            return Rotor("IV","ESOVPZJAYQUIRHXLNFTGKDCMWB", rotorPosition, 9, ringSetting)
        elif name == "V":
            return Rotor("V","VZBRGITYUPSDNHLXAWMJQOFECK", rotorPosition, 25, ringSetting)
        elif name == "VI":
            return Rotor("VI","JPGVOUMFYQBENHZRDKASXLICTW", rotorPosition, 0, ringSetting, True)
        elif name == "VII":
            return Rotor("VII","NZJHGRCXMYSWBOUFAIVLPEKQDT", rotorPosition, 0, ringSetting, True)
        elif name == "VIII":
                return Rotor("VIII","FKQHTLXOCBJSPDZRAMEWNIUYGV", rotorPosition, 0, ringSetting, True)
        else:
            return Rotor("Identity","ABCDEFGHIJKLMNOPQRSTUVWXYZ", rotorPosition, 0, ringSetting)


    def getName(self):
        return self.name
    

    def getPosition(self):
        return self.rotorPosition
    
    @staticmethod
    def decodeWiring(encoding):
        charWiring = encoding
        wiring = []
        for i in range(len(charWiring)):
            wiring.append(ord(charWiring[i]) - 65)
        return wiring

    @staticmethod
    def inverseWiring(wiring):
        inverse = list(range(len(wiring)))
        for i in range(len(wiring)):
            forward = wiring[i]
            inverse[forward] = i
        return inverse
    
    @staticmethod # protected static int
    def encipher(k, pos, ring, mapping):
        shift = pos - ring
        return (mapping[(k + shift + 26) % 26] - shift + 26) % 26
    

    def forward(self, c):
        return Rotor.encipher(c, self.rotorPosition, self.ringSetting, self.forwardWiring)
    

    def backward(self, c):
        return Rotor.encipher(c, self.rotorPosition, self.ringSetting, self.backwardWiring)
    

    def isAtNotch(self):
        if self.overrideIsAtNotch:
            self.rotorPosition == 12 or self.rotorPosition == 25
        else:
            return self.notchPosition == self.rotorPosition

    def turnover(self):
        self.rotorPosition = (self.rotorPosition + 1) % 26
    

class Enigma:
    def __init__(self,
        rotors,
        reflector,
        rotorPositions,
        ringSettings,
        plugboardConnections
    ):
        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
        # A B C D E F G H I J K  L  M  N  O  P  Q  R  S  T  U  V  W  X  Y  Z
        self.leftRotor = Rotor.Create(rotors[0], rotorPositions[0], ringSettings[0])
        self.middleRotor = Rotor.Create(rotors[1], rotorPositions[1], ringSettings[1])
        self.rightRotor = Rotor.Create(rotors[2], rotorPositions[2], ringSettings[2])
        self.reflector = Reflector.Create(reflector)
        self.plugboard = Plugboard(plugboardConnections)

    @staticmethod
    def init_by_key(key):
        return Enigma(key.rotors, "B", key.indicators, key.rings, key.plugboard)

    def rotate(self):
        # If middle rotor notch - double-stepping
        if (self.middleRotor.isAtNotch()):
            self.middleRotor.turnover()
            self.leftRotor.turnover()
        # If left-rotor notch
        elif (self.rightRotor.isAtNotch()):
            self.middleRotor.turnover()

        # Increment right-most rotor
        self.rightRotor.turnover()

    def encrypt(self, c):
        self.rotate()

        # Plugboard in
        c = self.plugboard.forward(c)

        # Right to left
        c1 = self.rightRotor.forward(c)
        c2 = self.middleRotor.forward(c1)
        c3 = self.leftRotor.forward(c2)

        # Reflector
        c4 = self.reflector.forward(c3)

        # Left to right
        c5 = self.leftRotor.backward(c4)
        c6 = self.middleRotor.backward(c5)
        c7 = self.rightRotor.backward(c6)

        # Plugboard out
        c7 = self.plugboard.forward(c7)

        return c7
    
    def encrypt_ch(self, c):
        return chr(self.encrypt(ord(c) - 65) + 65)
    

    def encrypt_text(self, text):
        output = []
        for i in range(len(text)):
            output.append(self.encrypt_ch(text[i]))
        return output
   
if __name__ == "__main__":
    print(
"""This is an Enigma machine, written by Eric 2021
This code based on Mike Pound's github repository:
https://github.com/mikepound/enigma
It is rewritten in pure Python code

Let's set the key for enigma machine as followings:
e = Engima(
    ["VII", "V", "IV"], "B", 
    [10,5,12], [1,2,3],
    "AD FT WH JO PN"
    )

Please input your text to process: """)

    e = Enigma(
        ["VII", "V", "IV"], 
        "B", 
        [10,5,12], 
        [1,2,3], 
        "AD FT WH JO PN"
    )

    #text = "HELLO WORLD"
    text = input().upper()
    print("\nFollowing is the output from Enigma Machine:")
    texts = text.split(" ")
    for t in texts:
        print(t, "".join(e.encrypt_text(t)))


