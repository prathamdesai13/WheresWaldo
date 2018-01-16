import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * Created by Antonio on 2018-01-03.
 */
public class ImageProcessor extends JPanel {

    static final String path = "..";

    static int desiredWidth = 35, desiredHeight = 35;
    static int windowScale = 40;
    static boolean zoom = true;

    static File[] waldos = new File(path+"/Test Waldos/").listFiles();//{"C:\\Users\\Antonio\\OneDrive - University of Waterloo\\Projects\\PycharmProjects\\WheresWaldo\\Maps\\4.png"};//
    static int waldo = -1;

    static String outputPath = path+"/Cropped Waldos/Waldos 35x35/";

    public static void main(String[] args){
        ImageProcessor ip = new ImageProcessor();
        JFrame frame = new JFrame("Waldo Image Processor by Antonio Kim");
        frame.add(ip);
        frame.setLocationRelativeTo(null);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setVisible(true);
        frame.setSize(desiredWidth*windowScale, desiredHeight*windowScale);
        ip.requestFocus();
    }

    int imageX = 0, imageY = 0, imageStep = 1;
    BufferedImage bi = null;
    public ImageProcessor(){
        this.setSize(desiredWidth*windowScale, desiredHeight*windowScale);
        switchWaldo();
        repaint();
        addKeyListener(new KeyListener() {
            @Override
            public void keyTyped(KeyEvent e) {}

            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_ADD){
                    imageStep *= 2;
                    System.out.println("ImageStep:   "+imageStep);
                }
                else if (e.getKeyCode() == KeyEvent.VK_SUBTRACT){
                    imageStep /= 2;
                    if (imageStep == 0) imageStep = 1;
                    System.out.println("ImageStep:   "+imageStep);
                }
                else if (e.getKeyCode() == KeyEvent.VK_UP){
                    if (e.isControlDown()){
                        moveImageToTop();
                    }
                    else {
                        moveImageUp();
                    }
                    repaint();
                }
                else if (e.getKeyCode() == KeyEvent.VK_DOWN){
                    if (e.isControlDown()){
                        moveImageToBottom();
                    }
                    else {
                        moveImageDown();
                    }
                    repaint();
                }
                else if (e.getKeyCode() == KeyEvent.VK_LEFT){
                    if (e.isControlDown()){
                        moveImageToLeftSide();
                    }
                    else {
                        moveImageLeft();
                    }
                    repaint();
                }
                else if (e.getKeyCode() == KeyEvent.VK_RIGHT){
                    if (e.isControlDown()){
                        moveImageToRightSide();
                    }
                    else {
                        moveImageRight();
                    }
                    repaint();
                }
                else if (e.isControlDown() && e.getKeyCode() == KeyEvent.VK_SPACE){
                    centerImage();
                    repaint();
                }
            }

            @Override
            public void keyReleased(KeyEvent e) {
                if (zoom && e.isControlDown() && e.getKeyCode() == KeyEvent.VK_S){
                    saveWaldo();
                }
                else if (e.isControlDown() && e.getKeyCode() == KeyEvent.VK_Z){
                    zoom = !zoom;
                    System.out.println("Zoom turned "+(zoom ? "on" : "off"));
                    repaint();
                }
                else if (e.getKeyCode() == KeyEvent.VK_BACK_SPACE){
                    switchWaldo();
                }
            }
        });
    }

    public void moveImageUp(){
        imageY += windowScale*imageStep;
        if (imageY > 0) moveImageToTop();
    }
    public void moveImageDown(){
        imageY -= windowScale*imageStep;
        if (imageY < -windowScale*(bi.getHeight()-desiredHeight)) moveImageToBottom();
    }
    public void moveImageLeft(){
        imageX += windowScale*imageStep;
        if (imageX > 0) moveImageToLeftSide();
    }
    public void moveImageRight(){
        imageX -= windowScale*imageStep;
        if (imageX < -windowScale*(bi.getWidth()-desiredWidth)) moveImageToRightSide();
    }
    public void moveImageToTop(){
        imageY = 0;
    }
    public void moveImageToBottom(){
        imageY = -windowScale*(bi.getHeight()-desiredHeight);
    }
    public void moveImageToLeftSide(){
        imageX = 0;
    }
    public void moveImageToRightSide(){
        imageX = -windowScale*(bi.getWidth()-desiredWidth);
    }
    public void centerImage(){
        imageX = (getWidth()-bi.getWidth()*windowScale)/2;
        imageY = (getHeight()-bi.getHeight()*windowScale)/2;
    }
    public void saveWaldo(){
        try {
            String[] croppedWaldos = new File(outputPath).list();
            String outputFileName = "Waldo";
            NUM: for (int i = 1;; ++i){
                for (String name : croppedWaldos){
                    if (name.equalsIgnoreCase("Waldo"+i+".png")){
                        continue NUM;
                    }
                }
                outputFileName += i;
                break;
            }
            File outputfile = new File(outputPath+outputFileName+".png");
            System.out.println(-imageX/windowScale+", "+-imageY/windowScale+", "+desiredWidth+", "+desiredHeight);
            ImageIO.write(bi.getSubimage(-imageX/windowScale, -imageY/windowScale, desiredWidth, desiredHeight), "png", outputfile);

        } catch (IOException e1) {
            e1.printStackTrace();
        }
    }
    public void switchWaldo(){
        ++waldo;
        if (waldo < waldos.length){
            try{
                System.out.println("Switching to:   "+waldos[waldo].getCanonicalPath()+"    ("+waldo+")");
                bi = ImageIO.read(new FileInputStream(waldos[waldo].getAbsolutePath()));//
                imageX = 0;
                imageY = 0;
                imageStep = 1;
                windowScale = 35;
                repaint();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
        else{
            System.out.println("All Waldos cropped!");
            System.exit(0);
        }
    }

    @Override
    public void paintComponent(Graphics g){
        super.paintComponent(g);
        if (bi != null){
            if (zoom) {
                g.drawImage(bi, imageX, imageY, windowScale * bi.getWidth(), windowScale * bi.getHeight(), null);
            }
            else{
                g.drawImage(bi, 0, 0, getWidth(), getHeight(), null);
            }
        }
    }

}
