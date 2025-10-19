# Plant Identification API

## Installation

```bash
pip install flask python-dotenv tensorflow
```

## Usage

### 1. Start the API server

```bash
python api_server.py
```

Server will run on `http://localhost:5000`

### 2. Test with curl

```bash
# Single image
curl -X POST -F "images=@test_image.jpg" http://localhost:5000/identify

# Multiple images
curl -X POST -F "images=@image1.jpg" -F "images=@image2.jpg" http://localhost:5000/identify

# Health check
curl http://localhost:5000/health
```

### 3. Mobile App Integration

#### React Native Example:

```javascript
const identifyPlant = async (imageUris) => {
  const formData = new FormData();

  imageUris.forEach((uri, index) => {
    formData.append("images", {
      uri: uri,
      type: "image/jpeg",
      name: `plant_${index}.jpg`,
    });
  });

  try {
    const response = await fetch("http://YOUR_SERVER:5000/identify", {
      method: "POST",
      body: formData,
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    const result = await response.json();
    return result;
  } catch (error) {
    console.error("Error:", error);
  }
};
```

#### Flutter Example:

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<Map<String, dynamic>> identifyPlant(List<String> imagePaths) async {
  var request = http.MultipartRequest(
    'POST',
    Uri.parse('http://YOUR_SERVER:5000/identify')
  );

  for (String path in imagePaths) {
    request.files.add(await http.MultipartFile.fromPath('images', path));
  }

  var response = await request.send();
  var responseData = await response.stream.toBytes();
  var result = json.decode(String.fromCharCode(responseData));

  return result;
}
```

## Response Format

```json
{
  "success": true,
  "image_path": "temp_uploads/uuid_image.jpg",
  "predictions": [
    {
      "identifier": "selaginella_aristata",
      "confidence": 0.85,
      "confidence_percentage": 85.23
    },
    {
      "identifier": "s_engleri_hieron",
      "confidence": 0.12,
      "confidence_percentage": 12.45
    }
  ]
}
```
