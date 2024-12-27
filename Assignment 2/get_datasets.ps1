# PowerShell script to download datasets for Assignment 2

$DATASETS_DIR = "utils/datasets"
New-Item -ItemType Directory -Force -Path $DATASETS_DIR

Set-Location $DATASETS_DIR

# Get Stanford Sentiment Treebank
Invoke-WebRequest -Uri http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip -OutFile stanfordSentimentTreebank.zip

Expand-Archive -Path "stanfordSentimentTreebank.zip" -DestinationPath .
Remove-Item stanfordSentimentTreebank.zip