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

    static int desiredWidth = 31, desiredHeight = 31;
    static int windowScale = 35;
    static boolean zoom = true;

    static String[] waldos = new File(path+"/Hey-Waldo-master/64/waldo").list();
    static int waldo = 6; // Last Switched = 13_2_11.jpg

    public static void main(String[] args){
        ImageProcessor ip = new ImageProcessor();
        JFrame frame = new JFrame("Waldo Image Processor by Antonio Kim");
        frame.add(ip);
        frame.setLocationRelativeTo(null);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
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
                    System.out.println("ImageStep:   "+imageStep);
                }
                else if (e.getKeyCode() == KeyEvent.VK_UP){
                    imageY += windowScale*imageStep;
                    if (imageY > 0) imageY = 0;
                    repaint();
                }
                else if (e.getKeyCode() == KeyEvent.VK_DOWN){
                    imageY -= windowScale*imageStep;
                    if (imageY < -windowScale*(bi.getHeight()-desiredHeight)) imageY = -windowScale*(bi.getHeight()-desiredHeight);
                    repaint();
                }
                else if (e.getKeyCode() == KeyEvent.VK_LEFT){
                    imageX += windowScale*imageStep;
                    if (imageX > 0) imageX = 0;
                    repaint();
                }
                else if (e.getKeyCode() == KeyEvent.VK_RIGHT){
                    imageX -= windowScale*imageStep;
                    if (imageX < -windowScale*(bi.getWidth()-desiredWidth)) imageX = -windowScale*(bi.getWidth()-desiredWidth);
                    repaint();
                }
            }

            @Override
            public void keyReleased(KeyEvent e) {
                if (e.isControlDown() && e.getKeyCode() == KeyEvent.VK_S){
                    try {
                        String[] croppedWaldos = new File(path+"/Cropped Waldos").list();
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
                        File outputfile = new File(path+"/Cropped Waldos/"+outputFileName+".png");
                        System.out.println(-imageX/windowScale+", "+-imageY/windowScale+", "+desiredWidth+", "+desiredHeight);
                        ImageIO.write(bi.getSubimage(-imageX/windowScale, -imageY/windowScale, desiredWidth, desiredHeight), "png", outputfile);

                    } catch (IOException e1) {
                        e1.printStackTrace();
                    }
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

    public void switchWaldo(){
        ++waldo;
        if (waldo < waldos.length){
            try{
                System.out.println("Switching to:   "+waldos[waldo]);
                bi = ImageIO.read(new FileInputStream(path+"/Hey-Waldo-master/64/waldo/"+waldos[waldo]));
                imageX = 0;
                imageY = 0;
                imageStep = 1;
                windowScale = 35;
                repaint();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
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
