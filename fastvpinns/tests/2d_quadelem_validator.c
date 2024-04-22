#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>

// base function values
void C_Q_Q0_2D_Funct(double xi, double eta, double *values)
{
    values[0] = 1;
}

// values of the derivatives in xi direction
void C_Q_Q0_2D_DeriveXi(double xi, double eta, double *values)
{
    values[0] = 0;
}

// values of the derivatives in eta direction
void C_Q_Q0_2D_DeriveEta(double xi, double eta, double *values)
{
    values[0] = 0;
}
// values of the derivatives in xi-xi  direction
void C_Q_Q0_2D_DeriveXiXi(double xi, double eta, double *values)
{
    values[0] = 0;
}
// values of the derivatives in xi-eta direction
void C_Q_Q0_2D_DeriveXiEta(double xi, double eta, double *values)
{
    values[0] = 0;
}
// values of the derivatives in eta-eta direction
void C_Q_Q0_2D_DeriveEtaEta(double xi, double eta, double *values)
{
    values[0] = 0;
}

void C_Q_Q1_2D_Funct(double xi, double eta, double *values)
{
    values[0] = 0.25 * (1 - xi - eta + xi * eta);
    values[1] = 0.25 * (1 + xi - eta - xi * eta);
    values[2] = 0.25 * (1 - xi + eta - xi * eta);
    values[3] = 0.25 * (1 + xi + eta + xi * eta);
}

// values of the derivatives in xi direction
void C_Q_Q1_2D_DeriveXi(double xi, double eta, double *values)
{
    values[0] = 0.25 * (-1 + eta);
    values[1] = 0.25 * (1 - eta);
    values[2] = 0.25 * (-1 - eta);
    values[3] = 0.25 * (1 + eta);
}

// values of the derivatives in eta direction
void C_Q_Q1_2D_DeriveEta(double xi, double eta, double *values)
{
    values[0] = 0.25 * (-1 + xi);
    values[1] = 0.25 * (-1 - xi);
    values[2] = 0.25 * (1 - xi);
    values[3] = 0.25 * (1 + xi);
}
// values of the derivatives in xi-xi  direction
void C_Q_Q1_2D_DeriveXiXi(double xi, double eta, double *values)
{
    values[0] = 0;
    values[1] = 0;
    values[2] = 0;
    values[3] = 0;
}
// values of the derivatives in xi-eta direction
void C_Q_Q1_2D_DeriveXiEta(double xi, double eta, double *values)
{
    values[0] = 0.25;
    values[1] = -0.25;
    values[2] = -0.25;
    values[3] = 0.25;
}
// values of the derivatives in eta-eta direction
void C_Q_Q1_2D_DeriveEtaEta(double xi, double eta, double *values)
{
    values[0] = 0;
    values[1] = 0;
    values[2] = 0;
    values[3] = 0;
}

void C_Q_Q2_2D_Funct(double xi, double eta, double *values)
{
    double xi0 = 0.5 * xi * (xi - 1);
    double xi1 = 1 - xi * xi;
    double xi2 = 0.5 * xi * (xi + 1);
    double eta0 = 0.5 * eta * (eta - 1);
    double eta1 = 1 - eta * eta;
    double eta2 = 0.5 * eta * (eta + 1);

    values[0] = xi0 * eta0;
    values[1] = xi1 * eta0;
    values[2] = xi2 * eta0;
    values[3] = xi0 * eta1;
    values[4] = xi1 * eta1;
    values[5] = xi2 * eta1;
    values[6] = xi0 * eta2;
    values[7] = xi1 * eta2;
    values[8] = xi2 * eta2;
}

// values of the derivatives in xi direction
void C_Q_Q2_2D_DeriveXi(double xi, double eta, double *values)
{
    double xi0 = xi - 0.5;
    double xi1 = -2 * xi;
    double xi2 = xi + 0.5;
    double eta0 = 0.5 * eta * (eta - 1);
    double eta1 = 1 - eta * eta;
    double eta2 = 0.5 * eta * (eta + 1);

    values[0] = xi0 * eta0;
    values[1] = xi1 * eta0;
    values[2] = xi2 * eta0;
    values[3] = xi0 * eta1;
    values[4] = xi1 * eta1;
    values[5] = xi2 * eta1;
    values[6] = xi0 * eta2;
    values[7] = xi1 * eta2;
    values[8] = xi2 * eta2;
}

// values of the derivatives in eta direction
void C_Q_Q2_2D_DeriveEta(double xi, double eta, double *values)
{
    double xi0 = 0.5 * xi * (xi - 1);
    double xi1 = 1 - xi * xi;
    double xi2 = 0.5 * xi * (xi + 1);
    double eta0 = eta - 0.5;
    double eta1 = -2 * eta;
    double eta2 = eta + 0.5;

    values[0] = xi0 * eta0;
    values[1] = xi1 * eta0;
    values[2] = xi2 * eta0;
    values[3] = xi0 * eta1;
    values[4] = xi1 * eta1;
    values[5] = xi2 * eta1;
    values[6] = xi0 * eta2;
    values[7] = xi1 * eta2;
    values[8] = xi2 * eta2;
}
// values of the derivatives in xi-xi  direction
void C_Q_Q2_2D_DeriveXiXi(double xi, double eta, double *values)
{
    double eta0 = 0.5 * eta * (eta - 1);
    double eta1 = 1 - eta * eta;
    double eta2 = 0.5 * eta * (eta + 1);

    values[0] = eta0;
    values[1] = -2 * eta0;
    values[2] = eta0;
    values[3] = eta1;
    values[4] = -2 * eta1;
    values[5] = eta1;
    values[6] = eta2;
    values[7] = -2 * eta2;
    values[8] = eta2;
}
// values of the derivatives in xi-eta direction
void C_Q_Q2_2D_DeriveXiEta(double xi, double eta, double *values)
{
    double xi0 = xi - 0.5;
    double xi1 = -2 * xi;
    double xi2 = xi + 0.5;
    double eta0 = eta - 0.5;
    double eta1 = -2 * eta;
    double eta2 = eta + 0.5;

    values[0] = xi0 * eta0;
    values[1] = xi1 * eta0;
    values[2] = xi2 * eta0;
    values[3] = xi0 * eta1;
    values[4] = xi1 * eta1;
    values[5] = xi2 * eta1;
    values[6] = xi0 * eta2;
    values[7] = xi1 * eta2;
    values[8] = xi2 * eta2;
}

// values of the derivatives in eta-eta direction
void C_Q_Q2_2D_DeriveEtaEta(double xi, double eta, double *values)
{
    double xi0 = 0.5 * xi * (xi - 1);
    double xi1 = 1 - xi * xi;
    double xi2 = 0.5 * xi * (xi + 1);

    values[0] = xi0;
    values[1] = xi1;
    values[2] = xi2;
    values[3] = -2 * xi0;
    values[4] = -2 * xi1;
    values[5] = -2 * xi2;
    values[6] = xi0;
    values[7] = xi1;
    values[8] = xi2;
}



int main()
{

    // Query Values  ( 20 points on each direction to be Queried, So 400 points in total)
    double xi[] = {-1.0, -0.8947368421052632, -0.7894736842105263, -0.6842105263157895, -0.5789473684210527, -0.47368421052631576, -0.3684210526315789, -0.2631578947368421, -0.15789473684210525, -0.05263157894736836, 0.05263157894736836, 0.15789473684210525, 0.2631578947368421, 0.3684210526315789, 0.47368421052631576, 0.5789473684210527, 0.6842105263157895, 0.7894736842105263, 0.8947368421052632, 1.0};
    double eta[] = {-1.0, -0.8947368421052632, -0.7894736842105263, -0.6842105263157895, -0.5789473684210527, -0.47368421052631576, -0.3684210526315789, -0.2631578947368421, -0.15789473684210525, -0.05263157894736836, 0.05263157894736836, 0.15789473684210525, 0.2631578947368421, 0.3684210526315789, 0.47368421052631576, 0.5789473684210527, 0.6842105263157895, 0.7894736842105263, 0.8947368421052632, 1.0};

    int num_shape_functions = 1;

    // For Q0 element
    std::ofstream Q0("Q0_result.csv");
    for (int n = 0; n < num_shape_functions; n++)
    {
        for (int i = 0; i < 20; i++)
        {
            for (int j = 0; j < 20; j++)
            {
                double values[1];
                C_Q_Q0_2D_Funct(xi[i], eta[j], values);
                Q0 << values[0] << ",";
            }
        }
    }
    Q0 << "\n";

    // Q0 - X Derivative
    for (int n = 0; n < num_shape_functions; n++)
    {
        for (int i = 0; i < 20; i++)
        {
            for (int j = 0; j < 20; j++)
            {
                double values[1];
                C_Q_Q0_2D_DeriveXi(xi[i], eta[j], values);
                Q0 << values[0] << ",";
            }
        }
    }

    Q0 << "\n";

    // Q0 - Y Derivative
    for (int n = 0; n < num_shape_functions; n++)
    {
        for (int i = 0; i < 20; i++)
        {
            for (int j = 0; j < 20; j++)
            {
                double values[1];
                C_Q_Q0_2D_DeriveEta(xi[i], eta[j], values);
                Q0 << values[0] << ",";
            }
        }
    }

    Q0 << "\n";

    // Q0 - XX Derivative
    for (int n = 0; n < num_shape_functions; n++)
    {
        for (int i = 0; i < 20; i++)
        {
            for (int j = 0; j < 20; j++)
            {
                double values[1];
                C_Q_Q0_2D_DeriveXiXi(xi[i], eta[j], values);
                Q0 << values[0] << ",";
            }
        }
    }

    Q0 << "\n";

    // Q0 - XY Derivative
    for (int n = 0; n < num_shape_functions; n++)
    {
        for (int i = 0; i < 20; i++)
        {
            for (int j = 0; j < 20; j++)
            {
                double values[1];
                C_Q_Q0_2D_DeriveXiEta(xi[i], eta[j], values);
                Q0 << values[0] << ",";
            }
        }
    }

    Q0 << "\n";

    // Q0 - YY Derivative
    for (int n = 0; n < num_shape_functions; n++)
    {
        for (int i = 0; i < 20; i++)
        {
            for (int j = 0; j < 20; j++)
            {
                double values[1];
                C_Q_Q0_2D_DeriveEtaEta(xi[i], eta[j], values);
                Q0 << values[0] << ",";
            }
        }
    }

    Q0 << "\n";

    Q0.close();

    // For Q1 element
    num_shape_functions = 4;

    // For Q1 element
    std::ofstream Q1("Q1_result.csv");
    for (int i = 0; i < 20; i++)
    {
        for (int j = 0; j < 20; j++)
        {
            double values[4];
            C_Q_Q1_2D_Funct(xi[i], eta[j], values);
            for (int n = 0; n < num_shape_functions; n++)
            {
                Q1 << values[n] << ",";
            }
        }
    }
    Q1 << "\n";

    // Q1 - X Derivative
    for (int i = 0; i < 20; i++)
    {
        for (int j = 0; j < 20; j++)
        {
            double values[4];
            C_Q_Q1_2D_DeriveXi(xi[i], eta[j], values);
            for (int n = 0; n < num_shape_functions; n++)
            {
                Q1 << values[n] << ",";
            }
        }
    }
    Q1 << "\n";

    // Q1 - Y Derivative
    for (int i = 0; i < 20; i++)
    {
        for (int j = 0; j < 20; j++)
        {
            double values[4];
            C_Q_Q1_2D_DeriveEta(xi[i], eta[j], values);
            for (int n = 0; n < num_shape_functions; n++)
            {
                Q1 << values[n] << ",";
            }
        }
    }
    Q1 << "\n";

    // Q1 - XX Derivative
    for (int i = 0; i < 20; i++)
    {
        for (int j = 0; j < 20; j++)
        {
            double values[4];
            C_Q_Q1_2D_DeriveXiXi(xi[i], eta[j], values);
            for (int n = 0; n < num_shape_functions; n++)
            {
                Q1 << values[n] << ",";
            }
        }
    }
    Q1 << "\n";

    // Q1 - XY Derivative
    for (int i = 0; i < 20; i++)
    {
        for (int j = 0; j < 20; j++)
        {
            double values[4];
            C_Q_Q1_2D_DeriveXiEta(xi[i], eta[j], values);
            for (int n = 0; n < num_shape_functions; n++)
            {
                Q1 << values[n] << ",";
            }
        }
    }
    Q1 << "\n";

    // Q1 - YY Derivative
    for (int i = 0; i < 20; i++)
    {
        for (int j = 0; j < 20; j++)
        {
            double values[4];
            C_Q_Q1_2D_DeriveEtaEta(xi[i], eta[j], values);
            for (int n = 0; n < num_shape_functions; n++)
            {
                Q1 << values[n] << ",";
            }
        }
    }
    Q1 << "\n";

    Q1.close();

    num_shape_functions = 9;

    // For Q2 element
    std::ofstream Q2("Q2_result.csv");
    for (int i = 0; i < 20; i++)
    {
        for (int j = 0; j < 20; j++)
        {
            double values[9];
            C_Q_Q2_2D_Funct(xi[i], eta[j], values);
            for (int n = 0; n < num_shape_functions; n++)
            {
                Q2 << values[n] << ",";
            }
        }
    }
    Q2 << "\n";

    // Q2 - X Derivative
    for (int i = 0; i < 20; i++)
    {
        for (int j = 0; j < 20; j++)
        {
            double values[9];
            C_Q_Q2_2D_DeriveXi(xi[i], eta[j], values);
            for (int n = 0; n < num_shape_functions; n++)
            {
                Q2 << values[n] << ",";
            }
        }
    }
    Q2 << "\n";

    // Q2 - Y Derivative
    for (int i = 0; i < 20; i++)
    {
        for (int j = 0; j < 20; j++)
        {
            double values[9];
            C_Q_Q2_2D_DeriveEta(xi[i], eta[j], values);
            for (int n = 0; n < num_shape_functions; n++)
            {
                Q2 << values[n] << ",";
            }
        }
    }
    Q2 << "\n";

    // Q2 - XX Derivative
    for (int i = 0; i < 20; i++)
    {
        for (int j = 0; j < 20; j++)
        {
            double values[9];
            C_Q_Q2_2D_DeriveXiXi(xi[i], eta[j], values);
            for (int n = 0; n < num_shape_functions; n++)
            {
                Q2 << values[n] << ",";
            }
        }
    }
    Q2 << "\n";

    // Q2 - XY Derivative
    for (int i = 0; i < 20; i++)
    {
        for (int j = 0; j < 20; j++)
        {
            double values[9];
            C_Q_Q2_2D_DeriveXiEta(xi[i], eta[j], values);
            for (int n = 0; n < num_shape_functions; n++)
            {
                Q2 << values[n] << ",";
            }
        }
    }
    Q2 << "\n";

    // Q2 - YY Derivative
    for (int i = 0; i < 20; i++)
    {
        for (int j = 0; j < 20; j++)
        {
            double values[9];
            C_Q_Q2_2D_DeriveEtaEta(xi[i], eta[j], values);
            for (int n = 0; n < num_shape_functions; n++)
            {
                Q2 << values[n] << ",";
            }
        }
    }
    Q2 << "\n";

    Q2.close();


    return 0;
}