#include <iostream>
#include <vector>
// #include <opencv2/opencv.hpp>
#include <chrono>

int r = 1;
int appendedBits = 0;
// std::pair<std::vector<bool>, std::pair<int, int>> imageToBits(const cv::Mat& image) {
//     std::vector<bool> bits;
//     for (int i = 0; i < image.rows; i++) {
//         for (int j = 0; j < image.cols; j++) {
//             cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
//             for (int k = 0; k < 3; k++) {
//                 for (int l = 0; l < 8; l++) {
//                     bits.push_back((pixel[k] >> l) & 1);
//                 }
//             }
//         }
//     }
//     return {bits, {image.rows, image.cols}};
// }

std::vector<bool> stringToBoolVector(std::string input) {
    std::vector<bool> bits;
    for (char c : input) {
        for (int i = 0; i < 8; i++) {
            bits.push_back((c >> i) & 1);
        }
    }
    return bits;
}

std::string boolVectorToString(std::vector<bool> bits) {
    std::string output;
    for (int i = 0; i < bits.size(); i += 8) {
        char c = 0;
        for (int j = 0; j < 8; j++) {
            c |= bits[i + j] << j;
        }
        output.push_back(c);
    }
    return output;
}

std::vector<std::vector<bool>> generatorMatrix(int r, int m) {
    if (m == 0) {
        throw std::invalid_argument("m must be greater than 0");
    }
    if (r > m) {
        throw std::invalid_argument("r must be less than or equal to m");
    }
    if (r == 0) {
        return std::vector<std::vector<bool>>(1, std::vector<bool>(1 << m, true));
    }
    if (m == 1) {
        return {{1, 1}, {0, 1}};
    } else {
        std::vector<std::vector<bool>> smallerMatrix = generatorMatrix(r, m - 1);
        std::vector<std::vector<bool>> bottom = generatorMatrix(r - 1, m - 1);

        std::vector<std::vector<bool>> top;
        for (const auto& row : smallerMatrix) {
            std::vector<bool> newRow = row;
            newRow.insert(newRow.end(), row.begin(), row.end());
            top.push_back(newRow);
        }

        for (auto& row : bottom) {
            std::vector<bool> newRow(smallerMatrix[0].size(), false);
            newRow.insert(newRow.end(), row.begin(), row.end());
            top.push_back(newRow);
        }

        return top;
    }
}

std::vector<std::vector<bool>> splitVectorForEncoding(std::vector<bool> bits, int r, int m) {
    int chunkSize = m + 1;
    std::vector<std::vector<bool>> chunks;

    for (size_t i = 0; i < bits.size(); i += chunkSize) {
        std::vector<bool> chunk(bits.begin() + i, bits.begin() + std::min(bits.size(), i + chunkSize));
        while (chunk.size() < chunkSize) {
            chunk.push_back(0);
            appendedBits++;
        }
        chunks.push_back(chunk);
    }
    return chunks;
}

bool dotProductMod2(const std::vector<bool>& v1, const std::vector<bool>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same length");
    }
    bool result = 0;
    for (size_t i = 0; i < v1.size(); i++) {
        result ^= v1[i] & v2[i];
    }
    return result;
}

std::vector<bool> vectorByMatrixMod2(const std::vector<bool>& vector, const std::vector<std::vector<bool>>& matrix) {
    std::vector<bool> result;
    int matrixLength = matrix.size();
    for (size_t j = 0; j < matrix[0].size(); j++) {
        std::vector<bool> column;
        for (int i = 0; i < matrixLength; i++) {
            column.push_back(matrix[i][j]);
        }
        result.push_back(dotProductMod2(vector, column));
    }
    return result;
}

std::vector<bool> encode(std::vector<bool> bits, int r, int m) {
    std::vector<std::vector<bool>> matrix = generatorMatrix(r, m);
    std::vector<std::vector<bool>> chunks = splitVectorForEncoding(bits, r, m);

    std::vector<bool> encoded;
    for (const auto& chunk : chunks) {
        std::vector<bool> encodedChunk = vectorByMatrixMod2(chunk, matrix);
        encoded.insert(encoded.end(), encodedChunk.begin(), encodedChunk.end());
    }
    return encoded;
}

std::vector<std::vector<bool>> splitMessageForDecoding(std::vector<bool> message, int m, int& appendedBits) {
    std::vector<std::vector<bool>> chunks;
    appendedBits = 0;
    for (size_t i = 0; i < message.size(); i += m) {
        std::vector<bool> chunk(message.begin() + i, message.begin() + std::min(message.size(), i + m));
        while (chunk.size() < m) {
            chunk.push_back(0);
            appendedBits++;
        }
        chunks.push_back(chunk);
    }
    return chunks;
}

std::vector<int> convertToPm1(std::vector<bool> message) {
    std::vector<int> pm1;
    for (bool bit : message) {
        pm1.push_back(bit ? 1 : -1);
    }
    return pm1;
}

std::vector<std::vector<int>> generateUnitaryMatrix(int n) {
    if (n < 1) {
        throw std::invalid_argument("n must be greater than 0");
    }
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        matrix[i][i] = 1;
    }
    return matrix;
}

// Flattened Kronecker product function
std::vector<int> generateKroneckerProductFlat(const std::vector<int>& A, size_t aRows, size_t aCols,
                                              const std::vector<int>& B, size_t bRows, size_t bCols) {
    size_t resultRows = aRows * bRows;
    size_t resultCols = aCols * bCols;
    std::vector<int> result(resultRows * resultCols, 0);

    for (size_t i = 0; i < aRows; ++i) {
        for (size_t j = 0; j < aCols; ++j) {
            for (size_t k = 0; k < bRows; ++k) {
                for (size_t l = 0; l < bCols; ++l) {
                    size_t row = i * bRows + k;
                    size_t col = j * bCols + l;
                    result[row * resultCols + col] = A[i * aCols + j] * B[k * bCols + l];
                }
            }
        }
    }

    return result;
}
int getElement(const std::vector<int>& matrix, size_t row, size_t col, size_t cols) {
    return matrix[row * cols + col];
}

// Generate unitary matrix (flattened)
std::vector<int> generateUnitaryMatrixFlat(size_t n) {
    std::vector<int> unitaryMatrix(n * n, 0);
    for (size_t i = 0; i < n; ++i) {
        unitaryMatrix[i * n + i] = 1; // Diagonal elements are 1
    }
    return unitaryMatrix;
}

// generateHiM function with 1D approach
std::vector<int> generateHiMFlat(int i, int m) {
    size_t size1 = 1 << (m - i);
    size_t size2 = 1 << (i - 1);

    std::vector<int> I1 = generateUnitaryMatrixFlat(size1);
    std::vector<int> H = {1, 1, 1, -1}; // 2x2 Hadamard matrix (flattened)
    std::vector<int> HiM = generateKroneckerProductFlat(I1, size1, size1, H, 2, 2);
    std::vector<int> I2 = generateUnitaryMatrixFlat(size2);
    HiM = generateKroneckerProductFlat(HiM, size1 * 2, size1 * 2, I2, size2, size2);

    return HiM;
}

std::pair<int, int> findLargestComponentPosition(const std::vector<int>& vector) {
    int max_value = std::abs(vector[0]);
    int position = 0;
    int sign = (vector[0] > 0) ? 1 : -1;
    for (size_t i = 1; i < vector.size(); ++i) {
        if (std::abs(vector[i]) > max_value) {
            max_value = std::abs(vector[i]);
            sign = (vector[i] > 0) ? 1 : -1;
            position = i;
        }
    }
    return {position, sign};
}

std::vector<bool> intToUnpackedBitList(int n) {
    if (n < 0) {
        throw std::invalid_argument("n must be greater than or equal to 0");
    }
    int length = 0;
    int temp = n;
    while (temp > 0) {
        length++;
        temp >>= 1;
    }
    std::vector<bool> bitArray(length, false);
    for (int i = length - 1; i >= 0; --i) {
        bitArray[i] = n % 2;
        n /= 2;
    }
    return bitArray;
}

std::vector<int> dotProduct(std::vector<int> v1 , std::vector<int> v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same length");
    }
    std::vector<int> result;
    for (int i = 0; i < v1.size(); i++) {
        result.push_back(v1[i] * v2[i]);
    }
    return result;
}

// Perform vector-by-matrix multiplication with a flattened matrix
std::vector<int> vectorByMatrix(const std::vector<int>& vec, const std::vector<int>& matrix, size_t rows, size_t cols) {
    std::vector<int> result(rows, 0);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i] += vec[j] * matrix[i * cols + j];
        }
    }
    return result;
}

// Recursive Fast Hadamard Transform
void fastHadamardTransformRecursive(std::vector<int>& vec, size_t start, size_t end) {
    if (end - start == 1) return; // Base case: single element

    size_t mid = start + (end - start) / 2;
    fastHadamardTransformRecursive(vec, start, mid); // Transform the first half
    fastHadamardTransformRecursive(vec, mid, end);   // Transform the second half

    for (size_t i = start; i < mid; ++i) {
        int a = vec[i];
        int b = vec[i + (mid - start)];
        vec[i] = a + b;               // Combine results (sum)
        vec[i + (mid - start)] = a - b; // Combine results (difference)
    }
}

// Fast Hadamard Transform (driver)
std::vector<int> fastHadamardTransform(const std::vector<bool>& message, int m) {
    size_t N = 1 << m; // Length of the vector (2^m)
    std::vector<int> vector = convertToPm1(message);

    // Perform in-place recursive Hadamard Transform
    fastHadamardTransformRecursive(vector, 0, N);

    return vector;
}



std::vector<bool> reverseVector(const std::vector<bool>& vector) {
    std::vector<bool> reversed;
    for (int i = vector.size() - 1; i >= 0; --i) {
        reversed.push_back(vector[i]);
    }
    return reversed;
}

std::vector<bool> decode(std::vector<bool> message, int r, int m) {
    std::vector<int> transformedMessage;
    transformedMessage = fastHadamardTransform(message, m);
    auto [position, sign] = findLargestComponentPosition(transformedMessage);

    std::vector<bool> positionInBits = intToUnpackedBitList(position);
    positionInBits = reverseVector(positionInBits);
    while (positionInBits.size() < m + 1) {
        positionInBits.push_back(0);
    }
    if (sign == 1) {
        positionInBits.insert(positionInBits.begin(), 1);
        positionInBits.pop_back();
    } else {
        positionInBits.insert(positionInBits.begin(), 0);
        positionInBits.pop_back();
    }

    std::vector<bool> decoded = positionInBits;
    return decoded;
}


std::vector<bool> decodeChunks(std::vector<bool> message, int r, int m) {
    std::vector<bool> decoded;
    std::vector<std::vector<bool>> chunks = splitMessageForDecoding(message, 1 << m, appendedBits);
    for (const auto& chunk : chunks) {
        std::vector<bool> decodedChunk = decode(chunk, r, m);
        decoded.insert(decoded.end(), decodedChunk.begin(), decodedChunk.end());
    }

    return decoded;
}




int main() {
    std::cout << "Hello, World!" << std::endl;
    std::string input;
    std::cout << "Enter some input: ";
    std::getline(std::cin, input);
    std::cout << "You entered: " << input << std::endl;
    std::vector<bool> bits = stringToBoolVector(input);
    std::cout << "Bits: ";
    for (bool bit : bits) {
        std::cout << bit;
    }
    std::string output = boolVectorToString(bits);
    std::cout << "\nConverted back to string: " << output;
    
    // bits.clear();
    // for (int i = 0; i < 1000000; i++) {
    //     bits.push_back(rand() % 2);
    // }
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<bool> encodedBits = encode(bits, r, 25);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Encoding complete" << std::endl;
    std::cout << "Time taken to encode: " << elapsed.count() << " seconds" << std::endl;
    // std::cout << "Encoded bits: ";
    // for (bool bit : encodedBits) {
        // std::cout << bit;
    // }
    auto start2 = std::chrono::high_resolution_clock::now();
    std::vector<bool> decodedBits = decodeChunks(encodedBits, r, 25);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end2 - start2;
    std::cout << "Decoding complete" << std::endl;
    std::cout << "Time taken to decode: " << elapsed2.count() << " seconds" << std::endl;
    std::cout << "\nDecoded bits: ";
    for (bool bit : decodedBits) {
        std::cout << bit;
    }
    std::cout << boolVectorToString(decodedBits) << std::endl;

}