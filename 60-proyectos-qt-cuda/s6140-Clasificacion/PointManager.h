#ifndef POINTMANAGER_H
#define POINTMANAGER_H
#include <vector>
#include <random>

class PointManager {
    std::vector<float> m_x, m_x2, m_y;
public:
    PointManager() { generate(40); }
    void generate(int n) {
        m_x.clear(); m_x2.clear(); m_y.clear();
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dX(-1.0f, 1.0f);
        std::normal_distribution<float> dN(0.0f, 0.08f);
        for(int i=0; i<n; ++i) {
            float x = dX(gen);
            addPoint(x, (0.5f * x * x) + dN(gen));
        }
    }
    void addPoint(float x, float y) {
        m_x.push_back(x); m_x2.push_back(x*x); m_y.push_back(y);
    }
    const std::vector<float>& getX() const { return m_x; }
    const std::vector<float>& getX2() const { return m_x2; }
    const std::vector<float>& getY() const { return m_y; }
    int getCount() const { return (int)m_x.size(); }
};
#endif
