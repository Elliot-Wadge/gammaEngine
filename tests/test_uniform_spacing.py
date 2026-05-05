from gammaEngine.interpolation import uniform_spaced_points_on_sphere
import plotly.graph_objects as go

def test_uniform_spacing_speed(benchmark):
    R = 1
    N = 1000
    coords = uniform_spaced_points_on_sphere(N,R)

    def run_metric():
        reg = uniform_spaced_points_on_sphere(N,R)
        return reg

    reg = benchmark(run_metric)




if __name__ == '__main__':
    R = 0.02
    N = 13
    coords = uniform_spaced_points_on_sphere(N,R)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], mode='markers'))
    fig.show()

    
   

