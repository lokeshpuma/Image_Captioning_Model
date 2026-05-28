const facts = [
  "Encoder: Xception CNN (2048-D visual features)",
  "Decoder: LSTM sequence model",
  "Dataset: Flickr8k",
  "Inference: Greedy next-word decoding"
];

const factsList = document.getElementById("facts");

for (const fact of facts) {
  const item = document.createElement("li");
  item.textContent = fact;
  factsList.appendChild(item);
}
