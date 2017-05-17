The python implementation of the system described in Responsive Action-based Video Synthesis [http://visual.cs.ucl.ac.uk/pubs/actionVideo/]. This implementation is very much research code, meaning it is not particularly pretty nor very much tested. Use at your own risk :)

There's a lot of dependecies, but if you install Anaconda and deal with the errors from there, you should be fine. Feel free to drop me an email at corneliu.ilisescu[at]gmail.com if you need help and I'll do my best to help out.

To use, all you have to do is to run the SemanticControlGUI.py using python SemanticControlGUI.py and if the dependencies are all there, it should work out of the box. Once you have synthesised a new sequence, you should have a synthesised_sequence.npy in the folder you chose in the UI and you can use the script BlendAndRenderSynthesisedSequence.py to blend and render the sprites. Stay tuned for more comprehensive instructions on how to use the UI.

This software is provided as is without any warranties. You can use it, extend it, copy it and basically do whatever you want with it. The CMT tracker implementation is used as published by the original authors of Clustering of Static-Adaptive Correspondences for Deformable Object Tracking [https://www.gnebehay.com/cmt/]. I include their original readme/license so you must do the same.
