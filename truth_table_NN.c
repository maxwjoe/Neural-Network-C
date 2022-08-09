#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// EXPAND TO HIGHER TRAINING SETS AND INPUTS

// --- Simple Neural Network that learns boolean functions from a Truth Table ---
#define NUM_INPUT_NEURONS 4
#define NUM_HIDDEN_NEURONS 10
#define NUM_OUTPUT_NEURONS 7
#define NUM_TRAINING_SETS 10
#define NUM_EPOCHS 10000
#define LEARNING_RATE 0.2f

// --- Function Definitions ---

// init_weights : Initialises Network Weights to random values on the range [0,1]
double init_weights()
{
    return (double)rand() / (double)RAND_MAX;
}

// sigmoid : Neuron Activation Function
double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

// d_sigmoid : First Derivative of sigmoid function
double d_sigmoid(double x)
{
    return x * (1 - x);
}

// shuffle : Shuffles the dataset
void shuffle(int *data_set, size_t data_size)
{
    if (data_size > 1)
    {
        for (size_t i = 0; i < data_size - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (data_size - i));
            int t = data_set[j];
            data_set[j] = data_set[i];
            data_set[i] = t;
        }
    }
}

// NN_predict : Runs network on a given input
float NN_predict(double *inputs, double *output_layer, double *hidden_layer, double *hidden_layer_bias, double *output_layer_bias, double hidden_layer_weights[NUM_INPUT_NEURONS][NUM_HIDDEN_NEURONS], double output_layer_weights[NUM_HIDDEN_NEURONS][NUM_OUTPUT_NEURONS])
{
    // Compute Hidden Layer Activation
    for (int j = 0; j < NUM_HIDDEN_NEURONS; j++)
    {
        double activation = hidden_layer_bias[j];
        for (int k = 0; k < NUM_INPUT_NEURONS; k++)
        {
            activation += inputs[k] * hidden_layer_weights[k][j];
        }

        hidden_layer[j] = sigmoid(activation);
    }

    // Compute Output Layer Activation
    for (int j = 0; j < NUM_OUTPUT_NEURONS; j++)
    {
        double activation = output_layer_bias[j];
        for (int k = 0; k < NUM_HIDDEN_NEURONS; k++)
        {
            activation += hidden_layer[k] * output_layer_weights[k][j];
        }

        output_layer[j] = sigmoid(activation);
    }

    // Print Network Results
    printf("Input : ");
    for (int i = 0; i < NUM_INPUT_NEURONS; i++)
    {
        printf(" %d", (int)inputs[i]);
    }
    printf("       NN Output : ");
    for (int i = 0; i < NUM_OUTPUT_NEURONS; i++)
    {
        printf(" %f", output_layer[i]);
    }
    printf("       Thresholded Output : ");
    for (int i = 0; i < NUM_OUTPUT_NEURONS; i++)
    {
        printf(" %d", (int)round(output_layer[i]));
    }
    printf("\n");
}

// print_list : Prints an array in python list format
void print_list(double *list, int length)
{
    printf("[ ");
    for (int i = 0; i < length; i++)
    {
        printf((i < length - 1) ? "%f, " : "%f", list[i]);
    }
    printf(" ]\n");
}

// NN_Specs : Displays network specs
void NN_Specs(double *hidden_layer_bias, double *output_layer_bias, double hidden_layer_weights[NUM_INPUT_NEURONS][NUM_HIDDEN_NEURONS], double output_layer_weights[NUM_HIDDEN_NEURONS][NUM_OUTPUT_NEURONS])
{
    // --- Output Specs ---
    printf("\n--- Network Structure ---\n\n");
    printf("Input Neurons : %d\n", NUM_INPUT_NEURONS);
    printf("Hidden Layer Neurons : %d\n", NUM_HIDDEN_NEURONS);
    printf("Output Layer Neurons : %d\n", NUM_OUTPUT_NEURONS);
    printf("Epochs : %d\n", NUM_EPOCHS);
    printf("Number of Training Sets : %d\n", NUM_TRAINING_SETS);

    // --- Output Weights and Biases ---
    printf("\n--- Network Weights and Biases ---\n\n");
    printf("Hidden Layer Bias : \n");
    print_list(hidden_layer_bias, NUM_HIDDEN_NEURONS);
    printf("\nOutput Layer Bias : \n");
    print_list(output_layer_bias, NUM_OUTPUT_NEURONS);

    printf("\nHidden Layer Weights : \n");
    for (int i = 0; i < NUM_INPUT_NEURONS; i++)
    {
        printf("Hidden Neuron %d : ", i);
        print_list(hidden_layer_weights[i], NUM_HIDDEN_NEURONS);
    }

    printf("\nOutput Layer Weights : \n");
    for (int i = 0; i < NUM_HIDDEN_NEURONS; i++)
    {
        printf("Output Neuron %d : ", i);
        print_list(output_layer_weights[i], NUM_OUTPUT_NEURONS);
    }
    printf("\n\n");
}

int main()
{

    // --- Network Structure and Parameters ---
    const double learning_rate = 0.1f;

    double hidden_layer[NUM_HIDDEN_NEURONS];
    double output_layer[NUM_OUTPUT_NEURONS];

    double hidden_layer_bias[NUM_HIDDEN_NEURONS];
    double output_layer_bias[NUM_OUTPUT_NEURONS];

    double hidden_layer_weights[NUM_INPUT_NEURONS][NUM_HIDDEN_NEURONS];
    double output_layer_weights[NUM_HIDDEN_NEURONS][NUM_OUTPUT_NEURONS];

    // --- Setup Training Data (Seven-Segment Display) ---
    double training_inputs[NUM_TRAINING_SETS][NUM_INPUT_NEURONS] = {
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 1.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 1.0f},
        {0.0f, 1.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f, 1.0f}};
    double training_outputs[NUM_TRAINING_SETS][NUM_OUTPUT_NEURONS] = {{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f},
                                                                      {0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                                                                      {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f},
                                                                      {1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
                                                                      {0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f},
                                                                      {1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f},
                                                                      {1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                                                                      {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                                                                      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                                                                      {1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f}};

    // --- Initialise Weights and Biases ---
    for (int i = 0; i < NUM_INPUT_NEURONS; i++)
    {
        for (int j = 0; i < NUM_HIDDEN_NEURONS; j++)
        {
            hidden_layer_weights[i][j] = init_weights();
        }
    }

    for (int i = 0; i < NUM_HIDDEN_NEURONS; i++)
    {
        for (int j = 0; i < NUM_OUTPUT_NEURONS; j++)
        {
            output_layer_weights[i][j] = init_weights();
        }
    }

    for (int i = 0; i < NUM_OUTPUT_NEURONS; i++)
    {
        output_layer_bias[i] = init_weights();
    }

    printf("\n--- NN Before Training ---\n");
    NN_Specs(hidden_layer_bias, output_layer_bias, hidden_layer_weights, output_layer_weights);
    printf("Pre-Training Output : \n\n");

    // Run Network Before Training
    for (int i = 0; i < NUM_TRAINING_SETS; i++)
    {
        NN_predict(training_inputs[i], output_layer, hidden_layer, hidden_layer_bias, output_layer_bias, hidden_layer_weights, output_layer_weights);
    }

    // Train Network
    printf("\nTraining Network...\n\n");

    // --- Initialise Training Set Order ---
    int training_set_order[NUM_TRAINING_SETS];
    for (int i = 0; i < NUM_TRAINING_SETS; i++)
    {
        training_set_order[i] = i;
    }

    // --- Training Loop ---
    int epoch = 0;
    while (epoch < NUM_EPOCHS)
    {
        // Shuffle Training Data
        shuffle(training_set_order, NUM_TRAINING_SETS);

        // Cycle Through Training Data
        for (int x = 0; x < NUM_TRAINING_SETS; x++)
        {
            int i = training_set_order[x];

            // Forward Pass of Network

            // Compute Hidden Layer Activation
            for (int j = 0; j < NUM_HIDDEN_NEURONS; j++)
            {
                double activation = hidden_layer_bias[j];
                for (int k = 0; k < NUM_INPUT_NEURONS; k++)
                {
                    activation += training_inputs[i][k] * hidden_layer_weights[k][j];
                }

                hidden_layer[j] = sigmoid(activation);
            }

            // Compute Output Layer Activation
            for (int j = 0; j < NUM_OUTPUT_NEURONS; j++)
            {
                double activation = output_layer_bias[j];
                for (int k = 0; k < NUM_HIDDEN_NEURONS; k++)
                {
                    activation += hidden_layer[k] * output_layer_weights[k][j];
                }

                output_layer[j] = sigmoid(activation);
            }

            // BackPropagation

            // Compute Change in output weights
            double delta_output[NUM_OUTPUT_NEURONS];
            for (int j = 0; j < NUM_OUTPUT_NEURONS; j++)
            {
                double error = training_outputs[i][j] - output_layer[j];
                delta_output[j] = error * d_sigmoid(output_layer[j]);
            }

            // Compute change in hidden weights
            double delta_hidden[NUM_HIDDEN_NEURONS];
            for (int j = 0; j < NUM_HIDDEN_NEURONS; j++)
            {
                double error = 0.0f;
                for (int k = 0; k < NUM_OUTPUT_NEURONS; k++)
                {
                    error += delta_output[k] * output_layer_weights[j][k];
                }
                delta_hidden[j] = error * d_sigmoid(hidden_layer[j]);
            }

            // Apply changes in output weights
            for (int j = 0; j < NUM_OUTPUT_NEURONS; j++)
            {
                output_layer_bias[j] += delta_output[j] * LEARNING_RATE;

                for (int k = 0; k < NUM_HIDDEN_NEURONS; k++)
                {
                    output_layer_weights[k][j] += hidden_layer[k] * delta_output[j] * LEARNING_RATE;
                }
            }

            // Apply changes in hidden weights
            for (int j = 0; j < NUM_HIDDEN_NEURONS; j++)
            {
                hidden_layer_bias[j] += delta_hidden[j] * LEARNING_RATE;

                for (int k = 0; k < NUM_INPUT_NEURONS; k++)
                {
                    hidden_layer_weights[k][j] += training_inputs[i][k] * delta_hidden[j] * LEARNING_RATE;
                }
            }
        }

        epoch++;
    }

    NN_Specs(hidden_layer_bias, output_layer_bias, hidden_layer_weights, output_layer_weights);
    printf("Post Training Output : \n\n");
    // Run Network Prior to Training
    for (int i = 0; i < NUM_TRAINING_SETS; i++)
    {
        NN_predict(training_inputs[i], output_layer, hidden_layer, hidden_layer_bias, output_layer_bias, hidden_layer_weights, output_layer_weights);
    }

    return 0;
}