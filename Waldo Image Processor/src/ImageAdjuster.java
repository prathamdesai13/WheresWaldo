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
public class ImageAdjuster extends JPanel {

    static final String path = "..";

    static int desiredWidth = 32, desiredHeight = 32;
    static int windowScale = 35;
    static boolean zoom = true;

    static File[] waldos = new File(path+"/Cropped Waldos/Waldos 35x35").listFiles(); //{"C:\\Users\\Antonio\\OneDrive - University of Waterloo\\Projects\\PycharmProjects\\WheresWaldo\\Maps\\4.png"};//
    static int waldo = -1; // Last Switched = 9_0_10.jpg

    public static void main(String[] args){
        ImageAdjuster ip = new ImageAdjuster();
        while (waldo < waldos.length){
            ip.switchWaldo();
            ip.saveWaldo();
            ip.moveImageToRightSide();
            ip.saveWaldo();
            ip.moveImageToBottom();
            ip.saveWaldo();
            ip.moveImageToLeftSide();
            ip.saveWaldo();
            ip.imageY += windowScale*ip.imageStep*1.5;
            ip.imageX -= windowScale*ip.imageStep*1.5;
            ip.saveWaldo();
        }
    }

    int imageX = 0, imageY = 0, imageStep = 1;
    BufferedImage bi = null;
    public ImageAdjuster(){
        this.setSize(desiredWidth*windowScale, desiredHeight*windowScale);
        //switchWaldo();
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
                    moveImageUp();
                    repaint();
                }
                else if (e.getKeyCode() == KeyEvent.VK_DOWN){
                    moveImageDown();
                    repaint();
                }
                else if (e.getKeyCode() == KeyEvent.VK_LEFT){
                    moveImageLeft();
                    repaint();
                }
                else if (e.getKeyCode() == KeyEvent.VK_RIGHT){
                    moveImageRight();
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
    public void saveWaldo(){
        try {
            String[] croppedWaldos = new File(path+"/Cropped Waldos/Waldos 30x30").list();
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
            File outputfile = new File(path+"/Cropped Waldos/Waldos 30x30/"+outputFileName+".png");
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
