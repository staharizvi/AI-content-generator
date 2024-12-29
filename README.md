# AI Content Generator

This project is an AI-powered text generation tool that utilizes the **Mistral-7B-Instruct** model to create content for various use cases such as blog posts, product descriptions, social media posts, and emails. The tool leverages the Hugging Face `transformers` library for seamless integration and content generation.

---

## Features

- **Customizable Content Generation**: Generate content tailored to specific topics, sentiments, and formats.
- **Pre-defined Templates**: Choose from multiple content types like blog posts, product descriptions, social media posts, and emails.
- **Advanced Generation Settings**: Control the length, tone, temperature, and other parameters for precise content generation.
- **Content Management**: Save generated outputs with metadata for future reference.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- GPU (recommended for faster performance)
- Install required Python libraries:
  ```bash
  pip install transformers torch accelerate
  ```

## Usage

### 1. Configure Your Hugging Face Token
Add your Hugging Face token to the script or set it as an environment variable:

### 2. Run the Program
Execute the script:
```bash
python main.py
```

### 3. Follow On-Screen Instructions
The program will guide you through selecting a content type, entering parameters, and generating content.

---

## Content Types

### Blog Post
- **Parameters**:
  - `topic`: Main topic of the blog post.
  - `sentiment`: Tone of the content (e.g., professional, casual, enthusiastic).
  - `aspect`: Specific aspect to focus on.
- **Example Output**:
  ```json
  {
      "topic": "Artificial Intelligence",
      "sentiment": "enthusiastic",
      "aspect": "its impact on modern industries",
      "generated_content": [
          "Artificial intelligence (AI) is revolutionizing modern industries like never before! From automating routine tasks to providing insightful analytics, AI empowers businesses to innovate and thrive. This blog explores how AI is reshaping healthcare, finance, manufacturing, and more, offering exciting possibilities for the future."
      ]
  }
  ```

### Product Description
- **Parameters**:
  - `product`: Name of the product.
  - `sentiment`: Tone of the content (e.g., professional, casual, enthusiastic).
  - `features`: Key features to highlight.
- **Example Output**:
  ```json
  {
      "product": "Smart Fitness Watch",
      "sentiment": "professional",
      "features": "heart rate monitoring, sleep tracking, GPS, and waterproof design",
      "generated_content": [
          "Elevate your fitness journey with our Smart Fitness Watch. With advanced heart rate monitoring, precise sleep tracking, GPS for outdoor activities, and a waterproof design, this watch is your ultimate companion for achieving your fitness goals. Stay connected and motivated every step of the way!"
      ]
  }
  ```

### Social Media Post
- **Parameters**:
  - `topic`: Main topic of the post.
  - `sentiment`: Tone of the content (e.g., professional, casual, enthusiastic).
  - `platform`: Social media platform (e.g., Twitter, Instagram).
- **Example Output**:
  ```json
  {
      "topic": "National Robotics Week",
      "sentiment": "casual",
      "platform": "Twitter",
      "generated_content": [
          "Happy National Robotics Week! 🤖🎉 Dive into the world of robots and explore how they're shaping our future. From self-driving cars to smart assistants, the possibilities are endless! Share your favorite robotics innovation below. #RoboticsWeek #Tech"
      ]
  }
  ```

### Email
- **Parameters**:
  - `topic`: Main topic of the email.
  - `sentiment`: Tone of the content (e.g., formal, casual, professional).
  - `purpose`: Purpose of the email (e.g., invitation, announcement).
- **Example Output**:
  ```json
  {
      "topic": "Upcoming Webinar on AI Trends",
      "sentiment": "formal",
      "purpose": "inviting participants",
      "generated_content": [
          "Subject: Join Our Webinar on AI Trends Shaping 2024\n\nDear [Recipient],\n\nWe are excited to invite you to our upcoming webinar on 'AI Trends Shaping 2024'. Gain valuable insights from industry experts, explore innovative applications, and discover how AI is driving change across sectors. Don't miss this opportunity to stay ahead in the tech revolution!\n\nRegister now to secure your spot.\n\nBest regards,\n[Your Name/Organization]"
      ]
  }
  ```

---
---

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your changes.

---

## Contact
For questions or feedback, contact [PerceptiaAI](https://www.perceptiaai.com)
