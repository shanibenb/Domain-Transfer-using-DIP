import os
import time
import numpy as np
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
from util.measure_perceptual_loss import run_style_transfer

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.serial_batches = True   # no shuffle

    content_list = []
    style_list = []
    start_time = time.time()
    number_of_steps = 500

    for idxA in range(0, 50):  # Load from A dataset one image at a time
        model = create_model(opt)
        visualizer = Visualizer(opt)
        # create website
        web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

        opt.start = idxA

        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        dataset_size = len(data_loader)
        print('#training images = %d' % dataset_size)

        # model.reset_network(opt)
        total_steps = 0

        # for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            if total_steps > number_of_steps:
                break
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), idxA, save_result)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors_test(idxA, epoch_iter, errors, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(idxA, float(epoch_iter) / dataset_size, opt, errors)

            iter_data_time = time.time()

        print('End of image %d / %d \t Time Taken: %d sec' %
              (idxA, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

        visuals = model.get_current_visuals()
        # save images
        img_path = model.get_image_paths()
        print('%04d: process image... %s' % (idxA, img_path))
        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, index=idxA, split=0)

        # calc content loss and style loss
        content_img = (model.input_A * 0.5) + 0.5
        style_img = (model.input_B * 0.5) + 0.5
        input_img = (model.fake_B * 0.5) + 0.5
        content_new, style_new = run_style_transfer(content_img, style_img, input_img)
        content_list.append(content_new.item())
        style_list.append(style_new.item())

        webpage.save()
    end_time = time.time()
    print("time: " + str(end_time - start_time))

    print("mean content: " + str(np.nanmean(content_list)))
    print("mean style: " + str(np.nanmean(style_list)))
