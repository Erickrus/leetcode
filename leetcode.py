#@title 187. Repeated DNA Sequences
# https://leetcode.com/problems/repeated-dna-sequences/

class Solution:
  def findRepeatedDnaSequences(self, s: str) -> List[str]:
    occurrences, dup = {}, {}
    for i in range(9, len(s)):
      currSeq = s[i-9:i+1]
      if currSeq in occurrences:
        dup[currSeq] = 1
      else:
        occurrences[currSeq] = 1 
    return list(dup.keys())
