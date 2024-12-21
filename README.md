# Sentiment Analysis Project: Yelp Review Dataset

## **Project Task**
This project focuses on the **Sentiment Analysis** NLP task, which involves classifying text into sentiment-based categories. Using Yelp reviews, the goal is to predict ratings (from 1 to 5 stars) based on review content.

---

## **Dataset**
### **Yelp Review Full Dataset**
- **Description:** The dataset contains 650,000 Yelp reviews categorized into five-star ratings (1 to 5). Each review includes textual content and its corresponding rating.
- **Source:** [Yelp Review Full Dataset on HuggingFace](https://huggingface.co/datasets/Yelp/yelp_review_full)

### **Dataset Summary:**
- **Training Samples:** 650,000
- **Test Samples:** 50,000
- **Features:**
  - `text`: The review content.
  - `label`: The star rating (1 to 5).

---

## **Pre-trained Model**
### **Selected Model:** `distilbert-base-cased`
- **Why DistilBERT?**
  - DistilBERT is a lightweight, pre-trained transformer model that is faster and more resource-efficient than BERT while maintaining comparable performance.
  - It is ideal for tasks involving large datasets and limited computational resources.
- **HuggingFace Link:** [DistilBERT on HuggingFace](https://huggingface.co/distilbert-base-cased)

---

## **Performance Metrics**
### **Final Model Results:**
- **Accuracy:** 68%
- **Class-wise F1-Scores:**
  - Class 0: 0.79
  - Class 1: 0.63
  - Class 2: 0.62
  - Class 3: 0.60
  - Class 4: 0.75

### **Confusion Matrix:**
Below is a visualization of the confusion matrix showing model performance:

![Confusion Matrix](confusion_matrix.png)

---

## **Hyperparameters**
### **Optimized Hyperparameters:**
- **Learning Rate:** `2.745e-5`
- **Batch Size:** `8`
- **Epochs:** `2`
- **Weight Decay:** `0.026`
- **Gradient Accumulation Steps:** `4`

### **Why These Hyperparameters?**
- **Learning Rate:** A small value ensures stable convergence during training.
- **Batch Size:** A batch size of 8 balances memory usage and convergence speed.
- **Weight Decay:** Helps regularize the model and prevent overfitting.

---

## **Relevant Links**
- **Model on HuggingFace:** [Link](https://huggingface.co/your_model_link)
  
- **Dataset Link:** [Yelp Review Full Dataset](https://huggingface.co/datasets/Yelp/yelp_review_full)

---

## **Deployment Plan**
### **Tools/Processes for Deployment:**
- **Model Hosting:** Deploy the model on HuggingFace Spaces or AWS Sagemaker.
- **API Integration:** Use `FastAPI` or `Flask` to create REST APIs for real-time predictions.
- **Containerization:** Utilize Docker for consistent deployment environments.
- **Monitoring Tools:** Implement tools like Prometheus and Grafana to monitor latency and errors.

### **Deployment Process in Production:**
1. **Prepare the Model:** Save the fine-tuned model in HuggingFace-compatible format.
2. **Build APIs:** Develop REST endpoints for inference using `FastAPI`.
3. **Deploy on Cloud:** Host the model on cloud services (e.g., AWS, Google Cloud).
4. **Continuous Monitoring:** Monitor performance and implement A/B testing for updates.

### **Considerations and Concerns:**
- **Latency:** Ensure real-time predictions meet user expectations.
- **Scalability:** The system should handle increasing traffic.
- **Model Drift:** Regularly retrain the model with updated data.
- **Bias in Predictions:** Monitor outputs for unintended biases in classifications.

---

## **Visualizations**
1. **Class-wise F1-Scores:**
   - Bar chart showing F1-scores for each class.
2. **Confusion Matrix:**
   - Heatmap (visualized above).
3. **Accuracy vs Epochs:**
   - Line graph showing training accuracy over epochs.

### Example Visualization Code
```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]).plot(cmap="Blues")
plt.show()
# LLM Project

## Project Task
(fill in details about your chosen project)

## Dataset
(fill in details about the dataset you're using)

## Pre-trained Model
(fill in details about the pre-trained model you selected)

## Performance Metrics
(fill in details about your chosen metrics and results)

## Hyperparameters
(fill in details about which hyperparemeters you found most important/relevant while optimizing your model)

