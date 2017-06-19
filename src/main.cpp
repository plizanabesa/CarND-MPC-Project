#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"
//#include "matplotlibcpp.h"

/* References
 https://discussions.udacity.com/t/do-you-need-to-transform-coordiantes/256483/9
 */

//namespace plt = matplotlibcpp;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // Read telemetry data
            // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];
          double delta = j[1]["steering_angle"];
          delta = -delta;
          double accel = j[1]["throttle"];
            
          // Calculate steering angle and throttle using MPC. Both are in between [-1, 1].
          double steer_value;
          double throttle_value;
            
          // Transform reference trajectory points from map coordinate system to vehicle coordinate system
          Eigen::VectorXd ptsx_car(ptsx.size());
          Eigen::VectorXd ptsy_car(ptsy.size());
          for (int i = 0; i < ptsx.size(); i++ )
          {
            // Center points to the vehicle position given by (px,py)
            double shift_x = ptsx[i] - px;
            double shift_y = ptsy[i] - py;
                
            // Rotate points so the x axis is in the direction of the vehicle heading (psi)
            ptsx_car[i] = shift_x * cos(-psi) - shift_y * sin(-psi);
            ptsy_car[i] = shift_x * sin(-psi) + shift_y * cos(-psi);

          }
            
          // Fit third order polynomial to waypoints in vehicle coordinate system
          auto coeffs = polyfit(ptsx_car, ptsy_car, 3);
            
          // The car in vehicle coordinate system is always located in (0,0), and hence has a psi of 0 as well
          double px_car = 0;
          double py_car = 0;
          double psi_car = 0;
         
          // Convert speed from mph to m/s
            double v_ms = v * 0.44704;
            
          // Estimate CTE. The cross track error is calculated by evaluating at polynomial at x, f(x)
          // and subtracting y.
          double cte = polyeval(coeffs, px_car) - py_car;
            
          if (std::abs(cte) > 5){
            std::string reset_msg = "42[\"reset\",{}]";
            ws.send(reset_msg.data(), reset_msg.length(), uWS::OpCode::TEXT);
          }
            
          // Estimate the orientation error epsi.
          // Due to the sign starting at 0, the orientation error is -f'(x).
          // derivative of coeffs[0] + coeffs[1] * x + coeffs[2] * x^2 + coeffs[3] * x^3
          // -> coeffs[1] + 2 * coeffs[2] * x + 3 * coeffs[3] * x^2
          double poly_diff = coeffs[1] + 2 * coeffs[2] * px_car + 3 * coeffs[3] * pow(px_car, 2);
          double epsi = psi_car - atan(poly_diff);
            
          // Solve optimization model for predictive control
          Eigen::VectorXd state(6);
          // Add latency: predict state in  100ms
          double latency = 0.1;
          double Lf = 2.67;
          double f0 = coeffs[0] + coeffs[1] * px_car + coeffs[2] * px_car*px_car + coeffs[3] * px_car*px_car*px_car;
          double psides0 = atan(coeffs[1] + 2 * coeffs[2] * px_car + 3 * coeffs[3] * px_car*px_car);
          cte = f0 - py + v * sin(epsi) * latency;
          epsi = -psides0 * v * delta * latency / Lf;
          px_car = px_car + v * latency;
          psi = v * delta * latency / Lf;
          v = v + accel *latency;
          state << px_car, py_car, psi_car, v_ms, cte, epsi;
          auto vars = mpc.Solve(state, coeffs);
          
          json msgJson;
          // Get steering and throttle values. TODO: negative steering.
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].

          steer_value = -vars[0]/(deg2rad(25));
          throttle_value = vars[1];
          std::cout << "steer = " << steer_value << std::endl;
          std::cout << "throttle = " << throttle_value << std::endl;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;

          // Display the MPC predicted trajectory (Green line in simulator)
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;
          // Add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line
          for (int i = 2; i < vars.size(); i++){
            if (i % 2 == 0)
              mpc_x_vals.push_back(vars[i]);
            else
              mpc_y_vals.push_back(vars[i]);
          }
            
          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          // Display the waypoints/reference line (Yellow line in simulator)
          vector<double> next_x_vals;
          vector<double> next_y_vals;
           
          /*
          for (int i = 0; i < ptsx.size(); ++i) {
             next_x_vals.push_back(ptsx_car(i));
             next_y_vals.push_back(ptsy_car(i));
          }*/
        
          double poly_inc = 2.5;
          int num_points = 25;
          for (int i = 1; i < num_points; i++){
            next_x_vals.push_back(poly_inc*i);
            next_y_vals.push_back(polyeval(coeffs, poly_inc*i));
          }
            
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          // Debugging. Print controls and state values
          bool print_values = true;
          if (print_values){
              // Print waypoints
              for (int i = 0; i < next_x_vals.size(); i++) {
                  double x_t = next_x_vals[i];
                  double y_t = next_y_vals[i];
                  std::cout << " (x,y) reference  " << i << " : " << x_t << " , " << y_t << std::endl;
              }
              
              // Print predicted values
              for (int i = 0; i < mpc_x_vals.size(); i++) {
                  double x_t = mpc_x_vals[i];
                  double y_t = mpc_y_vals[i];
                  std::cout << " (x,y) predicted  " << i << " : " << x_t << " , " << y_t << std::endl;
              }
          }
            
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
