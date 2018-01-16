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
public class ImageAdjuster {

    static final String path = "..";

    static int desiredWidth = 32, desiredHeight = 32;
    static int windowScale = 35;

    static File[] waldos = new File(path+"/Cropped Waldos/Test Waldos 35x35/Waldos/").listFiles(); //{"C:\\Users\\Antonio\\OneDrive - University of Waterloo\\Projects\\PycharmProjects\\WheresWaldo\\Maps\\4.png"};//
    static int waldo = -1; // Last Switched = 9_0_10.jpg

    static String outputPath = path+"/Cropped Waldos/Test Waldos 32x32/Waldos/";

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
            ip.centerImage();
            ip.saveWaldo();
        }
    }

    int imageX = 0, imageY = 0;
    BufferedImage bi = null;

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
        imageX = (desiredWidth-bi.getWidth())*windowScale/2;
        imageY = (desiredHeight-bi.getHeight())*windowScale/2;
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
                windowScale = 35;
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
        else{
            System.out.println("All Waldos cropped!");
            System.exit(0);
        }
    }

}
