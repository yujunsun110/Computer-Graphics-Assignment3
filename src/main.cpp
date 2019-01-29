// This example is heavily based on the tutorial at https://open.gl 

// OpenGL Helpers to reduce the clutter
#include "Helpers.h"

// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>

// Linear Algebra Library
#include <Eigen/Core>
#include <Eigen/Dense>
#include <math.h> // use M_PI

// Timer
#include <chrono>

//string
#include <sstream>
#include <fstream>
#include <iostream>

#include <cctype> // use isdigit()
#include <vector>

using namespace std;


// View initialization
bool perspectiveActive = false;

float r = 5;
float l = -5;
float t = 3.2;
float b = -3.2;
float n = 1;
float f = 10;

// Model struct
struct Model
{
    // Contains the vertex positions
    VertexBufferObject VBO;
    // Contains the normals
    VertexBufferObject VBO_N;
    // Contains the flat shading normals
    VertexBufferObject VBO_N_FLAT;

    Eigen::MatrixXf V;
    Eigen::MatrixXf N;
    Eigen::MatrixXf N_FLAT;

    Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f scaling = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    bool selected = false;
    char shading = 0;
};


// Camera Position initialization
float camera_theta = 1.2f;
float camera_phi = 0.7f;
float camera_rho = 5.0f;

// Aspect ratio initialization
float aspect;




// https://en.cppreference.com/w/cpp/container/vector
std::vector<Model> objects; // vector of objects
int selectedModel = -1;

// https://stackoverflow.com/questions/349050/calculating-a-lookat-matrix
Eigen::Matrix4f lookAt(Eigen::Vector3f eye, Eigen::Vector3f at, Eigen::Vector3f up)
{
    Eigen::Vector3f Z = (eye - at).normalized();
    Eigen::Vector3f X = up.cross(Z).normalized();
    Eigen::Vector3f Y = Z.cross(X).normalized();
    // column wise!!
    Eigen::Matrix4f view;
    view << X(0),       Y(0),       Z(0),       0,
            X(1),       Y(1),       Z(1),       0,
            X(2),       Y(2),       Z(2),       0,
            -X.dot(eye), -Y.dot(eye), -Z.dot(eye), 1.0f;

    return view;
}

bool load_off_file(const char *filename, struct Model &object)
{
    ifstream in(filename, ios::in);
    if (!in)
    {
        std::cout << "Cannot open " << filename << std::endl;
        return false;
    }
    std::string line;
    //getline (istream& is, string& str);
    //Extracts characters from is and stores them into str until the \n is found 
    if (getline(in, line) && line.substr(0, 3) == "OFF")
    {
        int numberOfV, numberOfF;
        vector<Eigen::Vector3f> vs;//all vertices
        while (getline(in, line))
        {
            if (isdigit(line[0]))
            {
                istringstream s(line); // the line containing vertice and face number
                s >> numberOfV >> numberOfF;
                break;
            }
            else
                return false;
        }

        //scale the .off
        float scaleOfFile;
        for (int i = 0; i < numberOfV; i++)
        {
            if (getline(in, line))
            {
                istringstream s(line);
                float x, y, z;
                s >> x >> y >> z;
                Eigen::Vector3f v(x, y, z);
                if (v.norm() > scaleOfFile)
                    scaleOfFile = v.norm();
                vs.push_back(v);
            }
            else
                return false;
        }

        // center the model
        Eigen::Vector3f sumV(0.0f, 0.0f, 0.0f);
        for (int i = 0; i < vs.size(); i++)
            sumV += vs[i];
        Eigen::Vector3f averageV = sumV/vs.size();
        for (int i = 0; i < vs.size(); i++)
            vs[i] -= averageV;

        // scale the model
        object.scaling(0,0) /= scaleOfFile;
        object.scaling(1,1) /= scaleOfFile;
        object.scaling(2,2) /= scaleOfFile;

        // Compute the normals
        vector<vector<int>> triangles; // vector of faces
        vector<vector<Eigen::Vector3f>> normalOfFaces(numberOfV); // vector of normals
        for (int i = 0; i < numberOfF * 3; i += 3)
        {
            if (getline(in, line))
            {
                istringstream s(line);
                int points;
                s >> points;
                int a, b, c;
                s >> a >> b >> c;
                vector<int> newf = {a, b, c};
                triangles.push_back(newf);

                // https://stackoverflow.com/questions/1966587/given-3-pts-how-do-i-calculate-the-normal-vector
                Eigen::Vector3f normal = (vs[b] - vs[a]).cross(vs[c] - vs[a]);//calculate normal
                normalOfFaces[a].push_back(normal);
                normalOfFaces[b].push_back(normal);
                normalOfFaces[c].push_back(normal);
                
            }
        }

        vector<Eigen::Vector3f> ns;
        for (vector<Eigen::Vector3f> v : normalOfFaces)
        {
            Eigen::Vector3f sum(0.0f, 0.0f, 0.0f);
            for (int i = 0; i < v.size(); i++)
                sum = sum + v[i];
            Eigen::Vector3f average = sum/v.size();
            average.normalize();
            ns.push_back(average);
        }

        object.V.resize(3, numberOfF * 3);
        object.N.resize(3, numberOfF * 3);
        object.N_FLAT.resize(3, numberOfF * 3);

        int i = 0;
        for (vector<int> triangle : triangles)
        {
            int a = triangle[0];
            int b = triangle[1];
            int c = triangle[2];

            object.V.col(i++) << vs[a](0), vs[a](1), vs[a](2);
            object.V.col(i++) << vs[b](0), vs[b](1), vs[b](2);
            object.V.col(i++) << vs[c](0), vs[c](1), vs[c](2);

            
            object.N.col(i - 3) << ns[a](0), ns[a](1), ns[a](2);
            object.N.col(i - 2) << ns[b](0), ns[b](1), ns[b](2);
            object.N.col(i - 1) << ns[c](0), ns[c](1), ns[c](2);


            Eigen::Vector3f flatNormal = (ns[a] + ns[b] + ns[c]) / 3.0f;
            // For flat shading, three vertices of a triangle are the same
            object.N_FLAT.col(i - 3) << flatNormal(0), flatNormal(1), flatNormal(2);
            object.N_FLAT.col(i - 2) << flatNormal(0), flatNormal(1), flatNormal(2);
            object.N_FLAT.col(i - 1) << flatNormal(0), flatNormal(1), flatNormal(2);
        }

        return true;
    }

    return false;
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    // https://stackoverflow.com/questions/29153163/opengl-how-to-fit-a-window-on-resize
    int w = height * aspect; // w is width adjusted for aspect ratio, maintain aspect ratio
    int left = (width - w) / 2;
    glViewport(left, 0, w, height); 
    // https://www.khronos.org/registry/OpenGL-Refpages/es2.0/xhtml/glViewport.xml
}


void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    if (action == GLFW_RELEASE)
        return; // No action needed when the mouse is released

    // Get the position of the mouse in the window
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    // Get the size of the window
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // Camera position
    Eigen::Vector3f cameraPos(
        camera_rho * sin(camera_theta) * cos(camera_phi),
        camera_rho * sin(camera_theta) * sin(camera_phi),
        camera_rho * cos(camera_theta)
        );

    Eigen::Vector3f view = Eigen::Vector3f(0, 0, 0) - cameraPos;
    view.normalize();

    Eigen::Vector3f h = view.cross(Eigen::Vector3f(0, 0, 1));
    h.normalize();

    Eigen::Vector3f v = h.cross(view);
    v.normalize();

    // convert angle to radians
    float angle = 45.0f * M_PI / 180;
    float vLength = tan (angle / 2);
    float hLength = vLength * (width / height);

    v *= vLength;
    h *= hLength;

    // translate mouse coordinates so that the origin lies in the center of the view port
    xpos -= width / 2;
    ypos -= height / 2;

    // scale mouse coordinates so that half the view port width and height becomes 1
    xpos /= (width / 2);
    ypos /= (height / 2);


    // linear combination to compute intersection of picking ray with view port plane
    Eigen::Vector3f pos = cameraPos + view + float(xpos) * h + float(ypos) * v;

    // compute direction of picking ray by subtracting intersection point with camera position
    Eigen::Vector3f dir = pos - cameraPos;

    float mindistance = 100;
    int target = 0; // Initilize which object is selected

    for (int i = 0; i < objects.size(); i++)
    {
        Model object = objects[i];

        Eigen::Vector3f p0 = Eigen::Vector3f(object.translation(0,3), object.translation(1,3), object.translation(2,3));
        Eigen::Vector3f p1 = cameraPos;
        Eigen::Vector3f p2 = cameraPos + dir;

        float distance = ((p2 - p1).cross(p1 - p0)).norm();

        if (distance < mindistance)
        {
            target = i;
            mindistance = distance;
            cout << "Operate on " << target << endl;
        }

    }

    if (target != -1)
    {
        if (selectedModel != -1)
        {
            objects[selectedModel].selected = false;
        }
        objects[target].selected = true;
        selectedModel = target;
    }
    else
    {
        selectedModel = -1;
    }
}

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_RELEASE)
        return;

    struct Model object;
    switch (key)
    {
    // Object Loading
    case GLFW_KEY_1:
        cout << "Insert cube" << endl;
        object.VBO.init();
        object.VBO_N.init();
        object.VBO_N_FLAT.init();
        object.V.resize(3, 36);
        object.N.resize(3, 36);
        object.N_FLAT.resize(3, 36);
                
        object.V.col(0) << -0.5f, -0.5f, 0.5f;
        object.V.col(1) << -0.5f, 0.5f, 0.5f;
        object.V.col(2) << 0.5f, 0.5f, 0.5f;
        object.V.col(3) << 0.5f, 0.5f, 0.5f;
        object.V.col(4) << 0.5f, -0.5f, 0.5f;
        object.V.col(5) << -0.5f, -0.5f, 0.5f;

        object.V.col(6) << -0.5f, -0.5f, -0.5f;
        object.V.col(7) << -0.5f, 0.5f, -0.5f;
        object.V.col(8) << 0.5f, 0.5f, -0.5f;
        object.V.col(9) << 0.5f, 0.5f, -0.5f;
        object.V.col(10) << 0.5f, -0.5f, -0.5f;
        object.V.col(11) << -0.5f, -0.5f, -0.5f;

        object.V.col(12) << 0.5f, -0.5f, 0.5f;
        object.V.col(13) << 0.5f, -0.5f, -0.5f;
        object.V.col(14) << 0.5f, 0.5f, 0.5f;
        object.V.col(15) << 0.5f, 0.5f, 0.5f;
        object.V.col(16) << 0.5f, 0.5f, -0.5f;
        object.V.col(17) << 0.5f, -0.5f, -0.5f;

        object.V.col(18) << -0.5f, -0.5f, 0.5f;
        object.V.col(19) << -0.5f, -0.5f, -0.5f;
        object.V.col(20) << -0.5f, 0.5f, 0.5f;
        object.V.col(21) << -0.5f, 0.5f, 0.5f;
        object.V.col(22) << -0.5f, 0.5f, -0.5f;
        object.V.col(23) << -0.5f, -0.5f, -0.5f;

        object.V.col(24) << 0.5f, 0.5f, -0.5f;
        object.V.col(25) << 0.5f, 0.5f, 0.5f;
        object.V.col(26) << -0.5f, 0.5f, -0.5f;
        object.V.col(27) << -0.5f, 0.5f, -0.5f;
        object.V.col(28) << -0.5f, 0.5f, 0.5f;
        object.V.col(29) << 0.5f, 0.5f, 0.5f;

        object.V.col(30) << 0.5f, -0.5f, -0.5f;
        object.V.col(31) << 0.5f, -0.5f, 0.5f;
        object.V.col(32) << -0.5f, -0.5f, -0.5f;
        object.V.col(33) << -0.5f, -0.5f, -0.5f;
        object.V.col(34) << -0.5f, -0.5f, 0.5f;
        object.V.col(35) << 0.5f, -0.5f, 0.5f;

        // Six Faces of a Cube
        object.N
            << 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1,
            1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

        object.N_FLAT = object.N;


        object.VBO.update(object.V);
        object.VBO_N.update(object.N);
        object.VBO_N_FLAT.update(object.N_FLAT);
        objects.push_back(object);
        break;

    case GLFW_KEY_2:
        cout << "Insert bunny" << endl;
        object.VBO.init();
        object.VBO_N.init();
        object.VBO_N_FLAT.init();
        load_off_file("bunny.off", object);
        object.VBO.update(object.V);
        object.VBO_N.update(object.N);
        object.VBO_N_FLAT.update(object.N_FLAT);
        objects.push_back(object);
        break;

    case GLFW_KEY_3:
        cout << "Insert bumpy cube" << endl;
        object.VBO.init();
        object.VBO_N.init();
        object.VBO_N_FLAT.init();
        load_off_file("bumpy_cube.off", object);
        object.VBO.update(object.V);
        object.VBO_N.update(object.N);
        object.VBO_N_FLAT.update(object.N_FLAT);
        objects.push_back(object);
        break;

    // Camera Control: P/O/L/K/M/N
    case GLFW_KEY_P:
        camera_phi += 0.2f;
        cout << "Camera moves right." << endl;
        break;

    case GLFW_KEY_O:
        camera_phi -= 0.2f;
        cout << "Camera moves left." << endl;
        break;

    case GLFW_KEY_L:
        camera_theta += 0.2f;
        cout << "Camera moves up." << endl;
        break;
        
    case GLFW_KEY_K:
        camera_theta -= 0.2f;
        cout << "Camera moves down." << endl;
        break;

    case GLFW_KEY_M:
        camera_rho += 0.2f;
        cout << "Camera moves away from objects." << endl;
        break;

    case GLFW_KEY_N:
        camera_rho -= 0.2f;
        cout << "Camera moves closer to objects." << endl;
        break;

    // Object Control
    case GLFW_KEY_W:
        if (selectedModel != -1)
        {
            if (mods == GLFW_MOD_CONTROL)
            {
                Eigen::Matrix4f RX;
                float theta = 0.2f;
                RX << 
                    1,          0,          0,          0,
                    0,          cos(theta), -sin(theta),0,
                    0,          sin(theta), cos(theta), 0,
                    0,          0,          0,          1;
                objects[selectedModel].rotation *= RX;
                cout << "Rotate down." << endl;

            }
            else
                objects[selectedModel].translation(2,3) += 0.2f;
                cout << "Translation z+." << endl;
        }
        break;

    case GLFW_KEY_S:
        if (selectedModel != -1)
        {
            if (mods == GLFW_MOD_CONTROL)
            {
                Eigen::Matrix4f RX;
                float theta = -0.2f;
                RX << 
                    1,          0,          0,          0,
                    0,          cos(theta), -sin(theta),0,
                    0,          sin(theta), cos(theta), 0,
                    0,          0,          0,          1;
                objects[selectedModel].rotation *= RX;
                cout << "Rotate up." << endl;
            }
            else
                objects[selectedModel].translation(2,3) -= 0.2f;
                cout << "Translation z-." << endl;
        }
        break;

    case GLFW_KEY_A://roration
        if (selectedModel != -1)
        {
            if (mods == GLFW_MOD_CONTROL)
            {
                Eigen::Matrix4f RZ;
                float theta = -0.2f;
                RZ << 
                    cos(theta), -sin(theta),0,          0,
                    sin(theta), cos(theta), 0,          0,
                    0,          0,          1,          0,
                    0,          0,          0,          1;
                objects[selectedModel].rotation *= RZ;
                cout << "Rotate left." << endl;
            }
            else
                objects[selectedModel].translation(0,3) += 0.2f;
                cout << "Translation x+." << endl;
        }
        break;
        
    case GLFW_KEY_D:
        if (selectedModel != -1)
        {
            if (mods == GLFW_MOD_CONTROL)
            {
                Eigen::Matrix4f RZ;
                float theta = 0.2f;
                RZ << 
                    cos(theta), -sin(theta),0,          0,
                    sin(theta), cos(theta), 0,          0,
                    0,          0,          1,          0,
                    0,          0,          0,          1;
                objects[selectedModel].rotation *= RZ;
                cout << "Rotate right." << endl;

            }
            else
                objects[selectedModel].translation(0,3) -= 0.2f;
                cout << "Translation x-." << endl;
        }
        break;

    case GLFW_KEY_Z:
        if (selectedModel != -1)
        {
            if (mods == GLFW_MOD_CONTROL)
            {
                Eigen::Matrix4f RY;
                float theta = 0.2f;
                RY << 
                    cos(theta), 0, sin(theta), 0,
                    0,          1, 0,          0,
                    -sin(theta),0, cos(theta), 0,
                    0,          0, 0,          1;
                objects[selectedModel].rotation *= RY;
                cout << "Rotate clockwise." << endl;
            }
            else
                objects[selectedModel].translation(1,3) += 0.2f;
                cout << "Translation y+." << endl;
        }
        break;
        
    case GLFW_KEY_C: // y-
        if (selectedModel != -1)
        {
            if (mods == GLFW_MOD_CONTROL)
            {
                Eigen::Matrix4f RY;
                float theta = -0.2f;
                RY << 
                    cos(theta), 0, sin(theta), 0,
                    0,          1,          0, 0,
                    -sin(theta),0, cos(theta), 0,
                    0,          0,          0, 1;
                objects[selectedModel].rotation *= RY;
                cout << "Rotate counterclockwise." << endl;
            }
            else
                objects[selectedModel].translation(1,3) -= 0.2f;
                cout << "Translation y-." << endl;
        }
        break;

    case GLFW_KEY_Q:
        if (selectedModel != -1)
        {
            objects[selectedModel].scaling(0,0) *= 1.2f;
            objects[selectedModel].scaling(1,1) *= 1.2f;
            objects[selectedModel].scaling(2,2) *= 1.2f;
            cout << "Scale up." << endl;
        }
        break;

    case GLFW_KEY_E:
        if (selectedModel != -1)
        {
            objects[selectedModel].scaling(0,0) *= 0.8f;
            objects[selectedModel].scaling(1,1) *= 0.8f;
            objects[selectedModel].scaling(2,2) *= 0.8f;
            cout << "Scale down." << endl;
        }
        break;

    case GLFW_KEY_X:
        if (selectedModel != -1)
        {
            objects.erase(objects.begin() + selectedModel);
            selectedModel = -1;
            cout << "Delete object." << endl;
        }
        break;

    case GLFW_KEY_4:// wireframe
        if (selectedModel != -1)
        {
            objects[selectedModel].shading = 0;
            cout << "Switch shading mode." << endl;
        }
        break;
    case GLFW_KEY_5:// flat shading
        if (selectedModel != -1)
        {
            objects[selectedModel].shading = 1;
            cout << "Switch shading mode." << endl;
        }
        break;

    case GLFW_KEY_6:// phong shading
        if (selectedModel != -1)
        {
            objects[selectedModel].shading = 2;
            cout << "Switch shading mode." << endl;
        }
        break;

    case GLFW_KEY_7:// switch between ortho/persprctive mode
        perspectiveActive = false;
        cout << "Orthogonal projection activated." << endl;
        break;
    case GLFW_KEY_8:
        perspectiveActive = true;
        cout << "Perspective projection activated." << endl;

    default:
        break;
    }
    
}




int main(void)
{
    GLFWwindow* window;

    // Initialize the library
    if (!glfwInit())
        return -1;

    // Activate supersampling
    glfwWindowHint(GLFW_SAMPLES, 8);

    // Ensure that we get at least a 3.2 context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

    // On apple we have to load a core profile with forward compatibility
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    #ifndef __APPLE__
      glewExperimental = true;
      GLenum err = glewInit();
      if(GLEW_OK != err)
      {
        /* Problem: glewInit failed, something is seriously wrong. */
       fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
      }
      glGetError(); // pull and savely ignonre unhandled errors like GL_INVALID_ENUM
      fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
    #endif

    int major, minor, rev;
    major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
    minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
    rev = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
    printf("OpenGL version recieved: %d.%d.%d\n", major, minor, rev);
    printf("Supported OpenGL is %s\n", (const char*)glGetString(GL_VERSION));
    printf("Supported GLSL is %s\n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

    // Initialize the VAO
    // A Vertex Array Object (or VAO) is an object that describes how the vertex
    // attributes are stored in a Vertex Buffer Object (or VBO). This means that
    // the VAO is not the actual object storing the vertex data,
    // but the descriptor of the vertex data.
    VertexArrayObject VAO;
    VAO.init();
    VAO.bind();

    // Initialize the OpenGL Program
    // A program controls the OpenGL pipeline and it must contains
    // at least a vertex shader and a fragment shader to be valid
    Program program;
    const GLchar* vertex_shader = R"glsl(
	#version 150 core
	in vec3 position;
    in vec3 normal;

    out vec3 FragPos;
    out vec3 Normal;

    out vec3 viewPosition;

	uniform mat4 model;
	uniform mat4 view;
	uniform mat4 proj;
	void main() {
        FragPos = vec3(model * vec4(position, 1.0));
        Normal = mat3(transpose(inverse(model))) * normal;  
		
        gl_Position = proj * view * model * vec4(position, 1.0);
	}
)glsl";

const GLchar* fragment_shader = R"glsl(
	#version 150 core
	out vec4 outColor;

    in vec3 Normal;  
    in vec3 FragPos;  
  
    uniform vec3 lightPos; 
    uniform vec3 viewPos; 
    uniform vec3 lightColor;
    uniform vec3 objectColor;

    uniform bool flatShading;

	void main() {

        // ambient
        float ambientStrength = 0.2;
        vec3 ambient = ambientStrength * lightColor;
  	
        // diffuse 
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;
    
        // specular
        float specularStrength = 0.5;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);  
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * lightColor;  
        
        vec3 result = (ambient + diffuse + specular) * objectColor;
        outColor = vec4(result, 1.0);
	}
)glsl";
// Phong Shader: https://learnopengl.com/Lighting/Basic-Lighting


    // Compile the two shaders and upload the binary to the GPU
    // Note that we have to explicitly specify that the output "slot" called outColor
    // is the one that we want in the fragment buffer (and thus on screen)
    program.init(vertex_shader,fragment_shader,"outColor");
    program.bind();

    // The vertex shader wants the position of the vertices as an input.
    // The following line connects the VBO we defined above with the position "slot"
    // in the vertex shader
    // program.bindVertexAttribArray("position",VBO);

    // Register the keyboard callback
    glfwSetKeyCallback(window, key_callback);

    // Register the mouse callback
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // Update viewport
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Projection Matrix
    Eigen::Matrix4f projection_perspective, projection_ortho;

    // Apply projection matrix to corner points
 
    projection_ortho <<
        2/(r-l), 0, 0, 0,
        0, 2/(t-b), 0, 0,
        0, 0, 2/(f-n), 0,
        0, 0, -(n-f)/(n+f), 1;


    int width, height;
    glfwGetWindowSize(window, &width, &height);
    aspect = double(width)/double(height);
    cout << "Aspect = " << aspect << " with( "<<width<<", " <<height<<" )"<< endl;
    float theta = 45 * M_PI /180;

    // http://ivl.calit2.net/wiki/images/a/ae/04_ProjectionS15.pdf
    projection_perspective <<
    1/(aspect*tan(theta/2)), 0, 0, 0,
    0, 1/tan(theta/2), 0, 0,
    0, 0,  (n+f)/(n-f), -1,
    0, 0, (2*n*f)/(n-f), 0;
  



    projection_perspective.transposeInPlace();// If you want to replace a matrix by its own transpose, do NOT do this:
                                              //m = m.transpose(); // bug!!! caused by aliasing effect
    projection_ortho.transposeInPlace();

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window))
    {
        if (perspectiveActive)
            glUniformMatrix4fv(program.uniform("proj"), 1, GL_FALSE, projection_perspective.data());
        else
            glUniformMatrix4fv(program.uniform("proj"), 1, GL_FALSE, projection_ortho.data());

        // Bind your VAO (not necessary if you have only one)
        VAO.bind();

        // Bind your program
        program.bind();

        // Camera Position Update
        // lookat eye at up
        float x1 = camera_rho * sin(camera_theta) * cos(camera_phi);
        float y1 = camera_rho * sin(camera_theta) * sin(camera_phi);
        float z1 = camera_rho * cos(camera_theta);
        Eigen::Vector3f eye1;
        eye1 << x1,y1,z1;
        Eigen::Matrix4f view = lookAt(eye1, Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, 1.0f));

        view.transposeInPlace();

        glUniformMatrix4fv(program.uniform("view"), 1, GL_FALSE, view.data());

        // Phong
        glUniform3f(program.uniform("viewPos"),
                    camera_rho * sin(camera_theta) * cos(camera_phi),
                    camera_rho * sin(camera_theta) * sin(camera_phi),
                    camera_rho * cos(camera_theta));
        glUniform3f(program.uniform("lightColor"), 1.0f, 1.0f, 1.0f);
        glUniform3f(program.uniform("lightPos"), 0.5f, 1.0f, 1.0f);

        // Clear the framebuffer
        glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Enable depth test
        glEnable(GL_DEPTH_TEST);

        for (Model object : objects)
        {
            // put rotation matrix after the translation, then the rotation center will change by translate
            object.model = object.translation * object.rotation * object.scaling;
            glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, object.model.data());

            if (object.selected)
            {
                glUniform3f(program.uniform("objectColor"), 1.0f, 0.0f, 0.0f);
            }
            else
            {
                glUniform3f(program.uniform("objectColor"), 0.0f, 1.0f, 0.0f);
            }

            program.bindVertexAttribArray("normal", object.VBO_N);
            program.bindVertexAttribArray("position", object.VBO);
            if (object.shading == 0)
            {
                // wireframe shading
                glUniform3f(program.uniform("objectColor"), 0.0f, 0.0f, 0.0f);
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                glDrawArrays(GL_TRIANGLES, 0, object.V.cols());
            }
            else if (object.shading == 1)
            {
                glUniform1i(program.uniform("flatShading"), true);

                // flat shading
                program.bindVertexAttribArray("normal", object.VBO_N_FLAT);
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                glDrawArrays(GL_TRIANGLES, 0, object.V.cols());

                // draw wireframe
                glUniform3f(program.uniform("objectColor"), 0.0f, 0.0f, 1.0f);
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                glDrawArrays(GL_TRIANGLES, 0, object.V.cols());
            }
            else if (object.shading == 2)
            {
                // normal shading
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                glDrawArrays(GL_TRIANGLES, 0, object.V.cols());
            }
        }

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    // Deallocate opengl memory
    program.free();
    VAO.free();
    //VBO.free();

    // Deallocate glfw internals
    glfwTerminate();
    return 0;
}
