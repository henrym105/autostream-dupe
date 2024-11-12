# AI-Powered Live Stream Automatic Pan / Zoom 

This project uses AI to dynamically apply a digital pan and zoom to a video of any land field sport (basketball, soccer, football, hockey) so that the players are always in frame. This eliminates the need to have a someone manually operating the camera during a game, ideal for low budget productions like little league and high school sports. It leverages the YOLO model for human detection and OpenCV for video processing.

Created as an MVP that mimics the AutoStream feature in GameChanger's live stream app.

## For any inquiries or feedback, please contact:

- **Name:** Henry Marquardt  
- **Email:** [henrymarquardt1@gmail.com](mailto:henrymarquardt1@gmail.com)  
- **GitHub:** [henrym105](https://github.com/henrym105)
- **LinkedIn:** [Henry Marquardt](https://www.linkedin.com/in/henry-marquardt/)

## Project Setup

### 1. **Clone the repository:**
    ```sh
    git clone https://github.com/henrym105/autostream-dupe.git
    cd autostream-dupe
    ```

### 2. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

### 3. **(OPTIONAL) Download your own YouTube video clip:**
    ```sh
    python youtube_download.py
    ```
    - A small sample video `data/raw/example_video.mp4` is included in the repo for you convenience and will be used by default
    - in `youtube_download.py`, change the params in main function for url, desired_resolution, start and stop time (to create a short clip)


### 4. **Run the application:**
    ```sh
    python main.py
    ```
    - NOTE: will automatically save a local copy of the **Yolov11 nano** model weights from ultralytics, but you can change the model being used in `constants.py`


## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/fooBar`).
3. Commit your changes (`git commit -am 'Add some fooBar'`).
4. Push to the branch (`git push origin feature/fooBar`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [YOLO](https://pjreddie.com/darknet/yolo/)
- [OpenCV](https://opencv.org/)
- [Python](https://www.python.org/)
- Special thanks to the open-source community for their contributions.

## TODO

- [ ] Implement real-time streaming and processing.
- [ ] Enhance the user interface for easier configuration.
- [ ] Optimize the video processing pipeline for performance.
- [ ] Deploy to mobile and check performance.
- [ ] Add more documentation and examples.
