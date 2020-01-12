import json
import torch
import sys

distractor_index = int(sys.argv[1]) - 1

def get_ref():
  samples = []
  reference = {}
  with open("data/race_test.json", "r") as fh:
    for line in fh:
      sample = json.loads(line)
      samples.append(sample)
  for sample in samples:
    question_id = str(sample['id']['file_id']) \
                  + "_" \
                  + str(sample["id"]["question_id"])
    if question_id not in reference.keys():
      reference[question_id] = [sample['distractor']]
    else:
      reference[question_id].append(sample['distractor'])
  return reference

def get_hyp():
  hypothesis = torch.load("translated.hypothesis.pt")
  reference = get_ref()
  f = open("distractor_" + str(distractor_index + 1) + ".decodes", "w")
  g = open("distractor_" + str(distractor_index + 1) + ".targets", "w")
  a, b = [], []
  index = 0
  for question_id in hypothesis:
    for idx, pred in enumerate(hypothesis[question_id]):
      if idx != distractor_index:
        continue
      distractor = " ".join(pred).strip().replace("\n", "").strip()
      if len(distractor) > 0:
        for real in reference[question_id]:
          ref = " ".join(real).strip().replace("\n", "").strip()
          if len(ref) > 0:
            a.append(distractor)
            b.append(ref)
            assert len(a) == len(b)
            index += 1
      break
    if index > 100000:
      break
  for i in range(len(a)):
    f.write(a[i] + "\n")
    g.write(b[i] + "\n")
  f.close()
  g.close()

if __name__ == "__main__":
  get_hyp()
