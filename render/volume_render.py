
import yt


ds = yt.load("RD0023/RD0023")
sc = yt.create_scene(ds)

im, sc = yt.volume_render(ds)
cam = sc.camera
sc.camera.resolution = (800,800)
for i in cam.iter_zoom(100.0, 10):
    source = sc.sources['source_00']
    sc.add_source(source)
    tf = yt.ColorTransferFunction((-31, -26))
    tf.add_layers(5, w=0.01)
    source.set_transfer_function(tf)
    sc.render()
    sc.save("zoom_%04i.png" % i)





    #sc.camera.set_width(ds.quan(1, 'Mpc'))
#sc.show()
#sc.show(sigma_clip=4)
#sc.save('render_001',sigma_clip=4)



