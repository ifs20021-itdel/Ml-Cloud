const express = require('express');
const multer = require('multer');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const tf = require('@tensorflow/tfjs'); // Menggunakan tfjs murni untuk browser dan Node.js
const jpeg = require('jpeg-js'); // Untuk decoding gambar JPEG
const pngjs = require('pngjs').PNG; // Untuk decoding gambar PNG

const app = express();
const PORT = process.env.PORT || 4000;

// URL model Anda di GCS
const MODEL_URL = 'https://storage.googleapis.com/newmodel2/submissions-model/model.json';

// Configure multer untuk file upload
const upload = multer({
    limits: { fileSize: 1000000 }, // Maksimum 1MB
    fileFilter(req, file, cb) {
        if (!file.mimetype.startsWith('image/')) {
            return cb(new Error('Only image files are allowed!'));
        }

        const ext = path.extname(file.originalname).toLowerCase();
        if (!['.jpg', '.jpeg', '.png'].includes(ext)) {
            return cb(new Error('Only JPG, JPEG, and PNG files are allowed!'));
        }

        cb(null, true);
    }
});

// Variabel untuk menyimpan model
let loadedModel;

// Fungsi untuk memuat model
async function loadModel() {
    try {
        console.log(`Loading model from: ${MODEL_URL}`);
        loadedModel = await tf.loadGraphModel(MODEL_URL); // Load dari URL GCS
        console.log('Model loaded successfully');
    } catch (err) {
        console.error('Error loading model:', err);
        process.exit(1); // Keluar jika model gagal dimuat
    }
}

// Fungsi prediksi
async function predict(imageBuffer) {
    try {
        let inputTensor;

        // Cek tipe file berdasarkan header
        if (imageBuffer[0] === 137 && imageBuffer[1] === 80 && imageBuffer[2] === 78 && imageBuffer[3] === 71) { // Header file PNG
            const png = pngjs.sync.read(imageBuffer);
            const { width, height, data } = png;
            inputTensor = tf.tensor3d(data, [height, width, 4], 'int32'); // PNG memiliki RGBA
        } else {
            const { width, height, data } = jpeg.decode(imageBuffer, { useTArray: true });
            inputTensor = tf.tensor3d(data, [height, width, data.length / (width * height)], 'int32');
        }

        // Jika gambar memiliki 4 channel (RGBA), buang channel alpha
        if (inputTensor.shape[2] === 4) {
            inputTensor = tf.slice(inputTensor, [0, 0, 0], [-1, -1, 3]); // Ambil RGB saja
        }

        // Resize gambar ke 225x225
        const resizedImage = tf.image.resizeBilinear(inputTensor, [224, 224]);

        // Normalisasi ke rentang [0, 1]
        const tensor = resizedImage.expandDims(0).toFloat().div(255);

        // Lakukan prediksi dengan model
        const predictions = await loadedModel.predict(tensor).data();
        const isCancer = predictions[0] > 0.5; // Gunakan threshold 0.5
        return isCancer ? 'Cancer' : 'Non-cancer';
    } catch (err) {
        console.error('Error during prediction:', err);
        throw new Error('Error processing image for prediction.');
    }
}

// Endpoint prediksi
app.get('/', (req, res) => {
    res.send('Model is online');
});

app.post('/predict', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({
                status: 'fail',
                message: 'No file uploaded'
            });
        }

        // Ambil buffer gambar
        const imageBuffer = req.file.buffer;

        // Lakukan prediksi
        const result = await predict(imageBuffer);

        // Saran berdasarkan hasil prediksi
        const suggestion = result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.';
        const predictionId = uuidv4();
        const createdAt = new Date().toISOString();

        return res.json({
            status: 'success',
            message: 'Prediction completed successfully',
            data: {
                id: predictionId,
                result,
                suggestion,
                createdAt
            }
        });
    } catch (err) {
        console.error('Prediction error:', err);
        return res.status(500).json({
            status: 'fail',
            message: 'Terjadi kesalahan dalam melakukan prediksi',
            error: err.message
        });
    }
});

// Middleware untuk menangani error
app.use((err, req, res, next) => {
    console.error('Error:', err);
    res.status(400).json({
        status: 'fail',
        message: err.message
    });
});

// Jalankan server dan muat model
app.listen(PORT, async () => {
    await loadModel();
    console.log(`Server is running on port ${PORT} http://localhost:${PORT}`);
});
