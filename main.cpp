#include <iostream>
#include <cmath>
#include <iomanip>
#include <random>

using namespace std;


// ============================================================
// SIGMOID
// ============================================================
//
// sigma(x) = 1 / (1 + e^(-x))
//
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}


// ============================================================
// DERIVATIVE OF SIGMOID
// ============================================================
//
// If
//
// h = sigmoid(x)
//
// then
//
// sigmoid'(x) = h(1-h)
//
double sigmoidDerivativeFromOutput(double h)
{
    return h * (1.0 - h);
}


// ============================================================
// MAIN
// ============================================================

int main()
{
    cout << fixed << setprecision(8);

    // ========================================================
    // XOR DATASET
    // ========================================================

    double X[4][2] =
    {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    double Y[4] =
    {
        0.0,
        1.0,
        1.0,
        0.0
    };


    // ========================================================
    // RANDOM INITIALIZATION
    // ========================================================

    mt19937 generator(42);

    uniform_real_distribution<double> distribution(-1.0, 1.0);


    // ========================================================
    // PARAMETERS OF HIDDEN NEURON h1
    // ========================================================
    //
    // h1 = sigmoid(w11*x1 + w12*x2 + c1)
    //

    double w11 = distribution(generator);
    double w12 = distribution(generator);
    double c1  = distribution(generator);


    // ========================================================
    // PARAMETERS OF HIDDEN NEURON h2
    // ========================================================
    //
    // h2 = sigmoid(w21*x1 + w22*x2 + c2)
    //

    double w21 = distribution(generator);
    double w22 = distribution(generator);
    double c2  = distribution(generator);


    // ========================================================
    // OUTPUT PARAMETERS
    // ========================================================
    //
    // y_hat = sigmoid(v1*h1 + v2*h2 + c3)
    //

    double v1 = distribution(generator);
    double v2 = distribution(generator);
    double c3 = distribution(generator);


    // ========================================================
    // LEARNING RATE
    // ========================================================

    double eta = 0.5;


    // ========================================================
    // NUMBER OF TRAINING EPOCHS
    // ========================================================

    int epochs = 100000;


    // ========================================================
    // PRINT INITIAL NETWORK
    // ========================================================

    cout << "\nINITIAL PARAMETERS\n";
    cout << "===============================\n";

    cout << "w11 = " << w11 << endl;
    cout << "w12 = " << w12 << endl;
    cout << "c1  = " << c1  << endl;

    cout << endl;

    cout << "w21 = " << w21 << endl;
    cout << "w22 = " << w22 << endl;
    cout << "c2  = " << c2  << endl;

    cout << endl;

    cout << "v1  = " << v1 << endl;
    cout << "v2  = " << v2 << endl;
    cout << "c3  = " << c3 << endl;


    // ========================================================
    // TRAINING
    // ========================================================

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double totalLoss = 0.0;


        // ====================================================
        // LOOP OVER THE FOUR XOR POINTS
        // ====================================================

        for (int i = 0; i < 4; i++)
        {
            double x1 = X[i][0];
            double x2 = X[i][1];
            double y  = Y[i];


            // =================================================
            // FORWARD PROPAGATION
            // =================================================


            // -------------------------------------------------
            // FIRST HIDDEN NEURON
            // -------------------------------------------------
            //
            // a1 = w11*x1 + w12*x2 + c1
            //
            // h1 = sigmoid(a1)
            //

            double a1 =
                w11 * x1 +
                w12 * x2 +
                c1;

            double h1 = sigmoid(a1);


            // -------------------------------------------------
            // SECOND HIDDEN NEURON
            // -------------------------------------------------
            //
            // a2 = w21*x1 + w22*x2 + c2
            //
            // h2 = sigmoid(a2)
            //

            double a2 =
                w21 * x1 +
                w22 * x2 +
                c2;

            double h2 = sigmoid(a2);


            // -------------------------------------------------
            // OUTPUT NEURON
            // -------------------------------------------------
            //
            // a3 = v1*h1 + v2*h2 + c3
            //
            // y_hat = sigmoid(a3)
            //

            double a3 =
                v1 * h1 +
                v2 * h2 +
                c3;

            double y_hat = sigmoid(a3);


            // =================================================
            // LOSS
            // =================================================
            //
            // L = 1/2 * (y_hat - y)^2
            //

            double error = y_hat - y;

            double loss =
                0.5 *
                error *
                error;

            totalLoss += loss;


            // =================================================
            // BACKPROPAGATION
            // =================================================


            // -------------------------------------------------
            // DERIVATIVE OF LOSS WITH RESPECT TO OUTPUT
            // -------------------------------------------------
            //
            // dL/dy_hat = y_hat - y
            //

            double dL_dyhat =
                y_hat - y;


            // -------------------------------------------------
            // DERIVATIVE OF OUTPUT SIGMOID
            // -------------------------------------------------
            //
            // dy_hat/da3 =
            //
            // y_hat * (1 - y_hat)
            //

            double dyhat_da3 =
                sigmoidDerivativeFromOutput(y_hat);


            // -------------------------------------------------
            // CHAIN RULE
            // -------------------------------------------------
            //
            // dL/da3 =
            //
            // (y_hat-y)
            //
            // *
            //
            // y_hat(1-y_hat)
            //

            double dL_da3 =
                dL_dyhat *
                dyhat_da3;


            // =================================================
            // OUTPUT LAYER GRADIENTS
            // =================================================


            // -------------------------------------------------
            // dL/dv1
            // -------------------------------------------------
            //
            // =
            //
            // (y_hat-y)
            //
            // y_hat(1-y_hat)
            //
            // h1
            //

            double dL_dv1 =
                dL_da3 *
                h1;


            // -------------------------------------------------
            // dL/dv2
            // -------------------------------------------------

            double dL_dv2 =
                dL_da3 *
                h2;


            // -------------------------------------------------
            // dL/dc3
            // -------------------------------------------------

            double dL_dc3 =
                dL_da3;


            // =================================================
            // FIRST HIDDEN NEURON
            // =================================================


            // -------------------------------------------------
            // derivative of h1
            // -------------------------------------------------
            //
            // dh1/da1 = h1(1-h1)
            //

            double dh1_da1 =
                sigmoidDerivativeFromOutput(h1);


            // -------------------------------------------------
            // dL/da1
            // -------------------------------------------------
            //
            // dL/da1 =
            //
            // dL/da3
            //
            // *
            //
            // v1
            //
            // *
            //
            // h1(1-h1)
            //

            double dL_da1 =
                dL_da3 *
                v1 *
                dh1_da1;


            // -------------------------------------------------
            // dL/dw11
            // -------------------------------------------------
            //
            // =
            //
            // (y_hat-y)
            //
            // y_hat(1-y_hat)
            //
            // v1
            //
            // h1(1-h1)
            //
            // x1
            //

            double dL_dw11 =
                dL_da1 *
                x1;


            // -------------------------------------------------
            // dL/dw12
            // -------------------------------------------------

            double dL_dw12 =
                dL_da1 *
                x2;


            // -------------------------------------------------
            // dL/dc1
            // -------------------------------------------------

            double dL_dc1 =
                dL_da1;


            // =================================================
            // SECOND HIDDEN NEURON
            // =================================================

            double dh2_da2 =
                sigmoidDerivativeFromOutput(h2);


            double dL_da2 =
                dL_da3 *
                v2 *
                dh2_da2;


            double dL_dw21 =
                dL_da2 *
                x1;


            double dL_dw22 =
                dL_da2 *
                x2;


            double dL_dc2 =
                dL_da2;


            // =================================================
            // GRADIENT DESCENT
            // =================================================
            //
            // p <- p - eta * dL/dp
            //
            // IMPORTANT:
            //
            // all derivatives were calculated BEFORE
            // modifying the weights.
            // =================================================


            w11 -= eta * dL_dw11;

            w12 -= eta * dL_dw12;

            c1  -= eta * dL_dc1;


            w21 -= eta * dL_dw21;

            w22 -= eta * dL_dw22;

            c2  -= eta * dL_dc2;


            v1 -= eta * dL_dv1;

            v2 -= eta * dL_dv2;

            c3 -= eta * dL_dc3;
        }


        // ====================================================
        // PRINT LOSS DURING TRAINING
        // ====================================================

        if (epoch % 5000 == 0)
        {
            cout
                << "Epoch "
                << setw(6)
                << epoch
                << "    Loss = "
                << totalLoss / 4.0
                << endl;
        }
    }


    // ========================================================
    // FINAL PARAMETERS
    // ========================================================

    cout << "\n\nFINAL PARAMETERS\n";
    cout << "========================================\n";

    cout << "\nHidden neuron h1\n";

    cout << "w11 = " << w11 << endl;
    cout << "w12 = " << w12 << endl;
    cout << "c1  = " << c1  << endl;


    cout << "\nHidden neuron h2\n";

    cout << "w21 = " << w21 << endl;
    cout << "w22 = " << w22 << endl;
    cout << "c2  = " << c2  << endl;


    cout << "\nOutput neuron\n";

    cout << "v1 = " << v1 << endl;
    cout << "v2 = " << v2 << endl;
    cout << "c3 = " << c3 << endl;


    // ========================================================
    // FINAL DECISION LINES
    // ========================================================
    //
    // Hidden neuron 1:
    //
    // w11*x1 + w12*x2 + c1 = 0
    //
    //
    // Hidden neuron 2:
    //
    // w21*x1 + w22*x2 + c2 = 0
    //
    // ========================================================

    cout << "\n\nHIDDEN DECISION LINES\n";
    cout << "========================================\n";

    cout << "\nLine generated by h1:\n";

    cout
        << w11
        << " x1 + "
        << w12
        << " x2 + "
        << c1
        << " = 0\n";


    cout << "\nLine generated by h2:\n";

    cout
        << w21
        << " x1 + "
        << w22
        << " x2 + "
        << c2
        << " = 0\n";


    // ========================================================
    // OUTPUT DECISION LINE IN h1-h2 SPACE
    // ========================================================

    cout << "\n\nOUTPUT DECISION LINE IN (h1,h2) SPACE\n";
    cout << "========================================\n";

    cout
        << v1
        << " h1 + "
        << v2
        << " h2 + "
        << c3
        << " = 0\n";


    // ========================================================
    // FINAL PREDICTIONS
    // ========================================================

    cout << "\n\nFINAL PREDICTIONS\n";
    cout << "========================================\n";


    for (int i = 0; i < 4; i++)
    {
        double x1 = X[i][0];

        double x2 = X[i][1];

        double y = Y[i];


        // ----------------------------------------------------
        // HIDDEN 1
        // ----------------------------------------------------

        double a1 =
            w11 * x1 +
            w12 * x2 +
            c1;

        double h1 =
            sigmoid(a1);


        // ----------------------------------------------------
        // HIDDEN 2
        // ----------------------------------------------------

        double a2 =
            w21 * x1 +
            w22 * x2 +
            c2;

        double h2 =
            sigmoid(a2);


        // ----------------------------------------------------
        // OUTPUT
        // ----------------------------------------------------

        double a3 =
            v1 * h1 +
            v2 * h2 +
            c3;

        double y_hat =
            sigmoid(a3);


        int prediction =
            y_hat >= 0.5 ? 1 : 0;


        // ----------------------------------------------------
        // PRINT ALL MATHEMATICAL STEPS
        // ----------------------------------------------------

        cout << "\n----------------------------------------\n";

        cout
            << "Input: ("
            << x1
            << ", "
            << x2
            << ")\n";

        cout
            << "Target y = "
            << y
            << "\n";


        cout << "\nFIRST HIDDEN NEURON\n";

        cout
            << "a1 = "
            << w11 << "*" << x1
            << " + "
            << w12 << "*" << x2
            << " + "
            << c1
            << "\n";

        cout
            << "a1 = "
            << a1
            << "\n";

        cout
            << "h1 = sigmoid(a1) = "
            << h1
            << "\n";


        cout << "\nSECOND HIDDEN NEURON\n";

        cout
            << "a2 = "
            << w21 << "*" << x1
            << " + "
            << w22 << "*" << x2
            << " + "
            << c2
            << "\n";

        cout
            << "a2 = "
            << a2
            << "\n";

        cout
            << "h2 = sigmoid(a2) = "
            << h2
            << "\n";


        cout << "\nTRANSFORMED POINT\n";

        cout
            << "("
            << x1
            << ", "
            << x2
            << ")"
            << " -> "
            << "("
            << h1
            << ", "
            << h2
            << ")"
            << "\n";


        cout << "\nOUTPUT NEURON\n";

        cout
            << "a3 = "
            << v1 << "*" << h1
            << " + "
            << v2 << "*" << h2
            << " + "
            << c3
            << "\n";

        cout
            << "a3 = "
            << a3
            << "\n";


        cout
            << "y_hat = sigmoid(a3) = "
            << y_hat
            << "\n";


        cout
            << "Final prediction = "
            << prediction
            << "\n";
    }


    // ========================================================
    // FINAL LOSS
    // ========================================================

    double finalLoss = 0.0;


    for (int i = 0; i < 4; i++)
    {
        double x1 = X[i][0];

        double x2 = X[i][1];

        double y = Y[i];


        double h1 =
            sigmoid(
                w11 * x1 +
                w12 * x2 +
                c1
            );


        double h2 =
            sigmoid(
                w21 * x1 +
                w22 * x2 +
                c2
            );


        double y_hat =
            sigmoid(
                v1 * h1 +
                v2 * h2 +
                c3
            );


        double error =
            y_hat - y;


        finalLoss +=
            0.5 *
            error *
            error;
    }


    finalLoss /= 4.0;


    cout << "\n\nFINAL MEAN LOSS\n";
    cout << "========================================\n";

    cout
        << "Loss = "
        << finalLoss
        << endl;


    return 0;
}
